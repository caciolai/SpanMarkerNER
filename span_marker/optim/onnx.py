# General imports
import json
import random
import shutil
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import onnx
import onnxruntime as ort

# Modeling imports
import torch
from datasets import load_dataset

# Exporting imports
from optimum.exporters.onnx import export
from optimum.exporters.onnx.model_configs import (
    DummyTextInputGenerator,
    NormalizedTextConfig,
    OnnxConfig,
)
from optimum.onnxruntime.modeling_ort import ORTModelForTokenClassification

from span_marker.configuration import SpanMarkerConfig
from span_marker.optim import SpanMarkerDataCollator, SpanMarkerModel
from span_marker.output import SpanMarkerOutput
from span_marker.tokenizer import SpanMarkerTokenizer


class DummySpanMarkerInputGenerator(DummyTextInputGenerator):
    """
    Generates dummy SpanMarker inputs.
    """

    SUPPORTED_INPUT_NAMES = ("input_ids", "attention_mask", "position_ids", "num_marker_pairs", "num_words")

    def __init__(
        self,
        task: str,
        normalized_config: NormalizedTextConfig,
        batch_size: int,
        model_max_length: int,
        marker_max_length: int,
        **kwargs,
    ):
        sequence_length = model_max_length + 2 * marker_max_length
        super().__init__(
            task=task,
            normalized_config=normalized_config,
            batch_size=batch_size,
            sequence_length=sequence_length,
            **kwargs,
        )

        self.vocab_size = normalized_config.vocab_size
        self.model_max_length = model_max_length
        self.marker_max_length = marker_max_length

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):

        if input_name in ["input_ids", "position_ids"]:
            min_value = 0
            max_value = 2 if input_name != "input_ids" else self.vocab_size
            return self.random_int_tensor(
                shape=(self.batch_size, self.sequence_length), max_value=max_value, min_value=min_value, dtype=int_dtype
            )
        elif input_name == "attention_mask":
            return self.random_mask_tensor(
                shape=(self.batch_size, self.sequence_length, self.sequence_length),
                padding_side=self.padding_side,
                framework=framework,
                dtype="bool",
            )
        elif input_name == "num_marker_pairs":
            num_marker_pairs = self.marker_max_length // 2
            return torch.tensor([num_marker_pairs], dtype=torch.int64).expand(self.batch_size)
        elif input_name == "num_words":
            return self.random_int_tensor(
                shape=(self.batch_size,), max_value=self.model_max_length, min_value=1, dtype=int_dtype
            )
        else:
            raise RuntimeError(f"Don't know how to generate for `{input_name}`")


class SpanMarkerOnnxConfig(OnnxConfig):
    """
    Class for ONNX exportable model for SpanMarker
    """

    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig
    ATOL_FOR_VALIDATION = 1e-4
    DEFAULT_ONNX_OPSET = 17
    DYNAMIC_AXES = {0: "batch_size"}

    def __init__(
        self,
        config: "PretrainedConfig",
        int_dtype: str = "int64",
        float_dtype: str = "fp32",
        batch_size: int = 4,
        model_max_length: int = 128,
        marker_max_length: int = 64,
    ):

        super().__init__(
            config=config,
            task="token-classification",
            int_dtype=int_dtype,
            float_dtype=float_dtype,
            preprocessors=None,
            legacy=False,
        )

        self.batch_size = batch_size
        self.model_max_length = model_max_length
        self.marker_max_length = marker_max_length
        self.sequence_length = model_max_length + 2 * marker_max_length

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        """
        Dict containing the axis definition of the input tensors to provide to the model.

        Returns:
            `Dict[str, Dict[int, str]]`: A mapping of each input name to a mapping of axis position to the axes symbolic name.
        """
        return OrderedDict(
            {
                "input_ids": self.DYNAMIC_AXES,
                "attention_mask": self.DYNAMIC_AXES,
                "position_ids": self.DYNAMIC_AXES,
                "num_marker_pairs": self.DYNAMIC_AXES,
                "num_words": self.DYNAMIC_AXES,
            }
        )

    @property
    def outputs(self) -> Dict[str, Dict[int, str]]:
        """
        Dict containing the axis definition of the output tensors to provide to the model.

        Returns:
            `Dict[str, Dict[int, str]]`: A mapping of each output name to a mapping of axis position to the axes symbolic name.
        """
        return OrderedDict(
            {"logits": self.DYNAMIC_AXES, "num_marker_pairs": self.DYNAMIC_AXES, "num_words": self.DYNAMIC_AXES}
        )

    def generate_dummy_inputs(self, framework: str = "pt", **kwargs) -> Dict:
        dummy_inputs_generators = [
            DummySpanMarkerInputGenerator(
                self.task,
                self._normalized_config,
                batch_size=self.batch_size,
                model_max_length=self.model_max_length,
                marker_max_length=self.marker_max_length,
            )
        ]

        dummy_inputs = {}
        for input_name in self.inputs:
            input_was_inserted = False
            for dummy_input_gen in dummy_inputs_generators:
                if dummy_input_gen.supports_input(input_name):
                    dummy_inputs[input_name] = dummy_input_gen.generate(
                        input_name, framework=framework, int_dtype=self.int_dtype, float_dtype=self.float_dtype
                    )
                    input_was_inserted = True
                    break
            if not input_was_inserted:
                raise RuntimeError(
                    f'Could not generate dummy input for "{input_name}". Try adding a proper dummy input generator to '
                    "the model ONNX config."
                )
        return dummy_inputs


class ORTSpanMarkerModel(ORTModelForTokenClassification):
    """
    Custom ONNX Runtime model that adds marker-specific tensors to the output.
    """

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        num_marker_pairs: torch.Tensor,
        num_words: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> SpanMarkerOutput:
        # 1. Standard ORT inference (no private helpers)
        inputs = {
            "input_ids": input_ids.to(dtype=torch.int64),
            "attention_mask": attention_mask.to(dtype=torch.bool),
            "position_ids": position_ids.to(dtype=torch.int64),
            "num_marker_pairs.1": num_marker_pairs.to(dtype=torch.int64),
            "num_words.1": num_words.to(dtype=torch.int64),
        }

        # 2. Optional IO-binding (latest Optimum: self._use_io_binding)
        if self._use_io_binding:
            io_binding = self.session.io_binding()
            for name, tensor in inputs.items():
                if tensor is None:
                    continue
                tensor = tensor.cpu()
                io_binding.bind_input(
                    name=name,
                    device_type=tensor.device.type,
                    device_id=tensor.device.index or 0,
                    element_type=tensor.numpy().dtype,
                    shape=tensor.size(),
                    buffer_ptr=tensor.data_ptr(),
                )
            io_binding.bind_output("logits")
            self.session.run_with_iobinding(io_binding)
            logits = io_binding.copy_outputs_to_cpu()[0]
            logits = torch.from_numpy(logits).to(input_ids.device)
        else:
            ort_inputs = {k: v.detach().cpu().numpy() for k, v in inputs.items() if v is not None}
            logits = self.session.run(None, ort_inputs)[0]
            logits = torch.from_numpy(logits).to(input_ids.device)

        return SpanMarkerOutput(
            logits=logits,
            num_marker_pairs=num_marker_pairs.detach().cpu(),
            num_words=num_words.detach().cpu() if num_words is not None else None,
        )


class OnnxWrappedSpanMarkerModel(SpanMarkerModel):
    def __init__(self, config: SpanMarkerConfig, tokenizer: SpanMarkerTokenizer, onnx_path: Path, providers: List[str]):
        super().__init__(config=config, encoder=None)

        self.tokenizer = tokenizer
        self.data_collator = SpanMarkerDataCollator(
            tokenizer=self.tokenizer, marker_max_length=self.config.marker_max_length
        )

        session_opts = ort.SessionOptions()
        session_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        self.ort_model = ORTSpanMarkerModel.from_pretrained(
            onnx_path.parent,
            file_name=onnx_path.name,
            providers=providers,
            session_options=session_opts,
        )

        # Just to suppress warning
        self.to("cuda")

    def forward(self, *args, **kwargs):
        return self.ort_model(*args, **kwargs)
