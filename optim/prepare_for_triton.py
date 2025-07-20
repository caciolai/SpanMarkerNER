#!/usr/bin/env python3

# General imports
import argparse
import logging
import random
import shutil
from pathlib import Path
import time

import numpy as np
import onnx

# Modeling imports
import torch
import tritonclient.grpc.model_config_pb2 as mc
from datasets import load_dataset
from google.protobuf import text_format

# Exporting imports
from onnx import helper as onnx_helper
from optimum.exporters.onnx import export
from tritonclient.utils import np_to_triton_dtype

from span_marker.optim import SpanMarkerModel
from span_marker.optim.onnx import OnnxWrappedSpanMarkerModel, SpanMarkerOnnxConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export a SpanMarker PyTorch model to ONNX and validate the conversion"
    )
    parser.add_argument("--model_path", type=str, required=True, help="Path to the PyTorch SpanMarker model directory")
    parser.add_argument(
        "--output_path",
        type=str,
        default="./triton_model_repository",
        help="Root directory for the Triton model repository",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing ONNX model if it exists")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for ONNX export (default: 32)")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
        help="Device to use for model execution (default: cuda if available, else cpu)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility (default: 0)")
    parser.add_argument(
        "--dataset",
        type=str,
        default="conll2003",
        help="HuggingFace dataset name to use for validation (default: conll2003)",
    )
    parser.add_argument(
        "--dataset_split", type=str, default="test", help="Dataset split to use for validation (default: test)"
    )
    parser.add_argument(
        "--triton_model_name", type=str, default="spanmarker", help="Name of the model in the Triton repository"
    )
    parser.add_argument(
        "--triton_model_version", type=int, default=1, help="Version of the model in the Triton repository"
    )
    parser.add_argument(
        "--triton_backend",
        type=str,
        default="tensorrt",
        choices=["tensorrt", "onnxruntime"],
        help="Backend for the Triton model (default: tensorrt)",
    )
    parser.add_argument(
        "--gpu_ids", type=int, nargs="+", default=[0], help="List of GPU IDs to use for Triton instance group"
    )
    parser.add_argument("--skip_validation", action="store_true", help="Skip validation of the ONNX export")

    return parser.parse_args()


def io_to_tuple(vinfo):
    ttype = vinfo.type.tensor_type

    # Remove batch dimension
    dims = [d.dim_value for d in ttype.shape.dim][1:]

    np_dtype = onnx_helper.tensor_dtype_to_np_dtype(ttype.elem_type)
    dtype_enum = mc.DataType.Value(f"TYPE_{np_to_triton_dtype(np_dtype)}")
    return vinfo.name, dtype_enum, dims


def make_triton_config(model_name, backend, inputs, outputs, gpu_ids):
    cfg = mc.ModelConfig(name=model_name, backend=backend)

    # inputs
    for name, dtype_enum, dims in inputs:
        inp = cfg.input.add()
        inp.name = name
        inp.data_type = dtype_enum
        inp.dims.extend(dims)

    # outputs
    for name, dtype_enum, dims in outputs:
        out = cfg.output.add()
        out.name = name
        out.data_type = dtype_enum
        out.dims.extend(dims)

    # GPU placement
    grp = cfg.instance_group.add()
    grp.kind = mc.ModelInstanceGroup.KIND_GPU
    grp.count = 1
    grp.gpus.extend(gpu_ids)

    return text_format.MessageToString(cfg, use_short_repeated_primitives=True)


def main():
    args = parse_args()

    # Set random seeds for reproducibility
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Resolve model paths
    model_torch_path = Path(args.model_path).resolve().absolute()
    model_onnx_path = model_torch_path / "model.onnx"

    triton_model_dir = Path(args.output_path) / args.triton_model_name
    triton_model_version_dir = triton_model_dir / str(args.triton_model_version)
    triton_onnx_path = triton_model_version_dir / "model.onnx"
    triton_config_path = triton_model_dir / "config.pbtxt"

    # Check if Triton model directory already exists
    if triton_model_dir.exists():
        if args.overwrite:
            logging.info(f"Overwriting existing Triton model directory: {triton_model_dir}")
            shutil.rmtree(str(triton_model_dir))
        else:
            raise FileExistsError(
                f"Triton model directory already exists at {triton_model_dir}. Use --overwrite to replace it."
            )

    # Load the PyTorch model
    logging.info(f"Loading model from {model_torch_path}")
    model = SpanMarkerModel.from_pretrained(model_torch_path)
    model = model.eval()

    logging.info(f"Model parameters: {sum([p.numel() for p in model.parameters()])}")

    # Configure ONNX export
    onnx_dict_config = dict(
        batch_size=args.batch_size,
        model_max_length=model.config.model_max_length,
        marker_max_length=model.config.marker_max_length,
    )
    logging.info(f"ONNX config: {onnx_dict_config}")
    onnx_config = SpanMarkerOnnxConfig(model.config, **onnx_dict_config)

    # Export the model to ONNX
    logging.info(f"Exporting model to ONNX format at {model_onnx_path}...")
    _ = export(
        model=model,
        config=onnx_config,
        output=model_onnx_path,
        device=args.device,
        no_dynamic_axes=False,  # Keep dynamic axes for batch size if needed by Triton
        do_constant_folding=True,
    )

    # Validate the ONNX model
    if not args.skip_validation:
        logging.info("Checking ONNX model validity...")
        onnx.checker.check_model(str(model_onnx_path))
        logging.info("ONNX model is valid.")

        # Load dataset for validation
        logging.info(f"Loading {args.dataset} dataset for validation...")
        dataset = load_dataset(args.dataset, trust_remote_code=True)
        tokens = list(dataset[args.dataset_split]["tokens"])

        # Prepare models
        if args.device == "cuda":
            model = model.cuda()
            providers = ["CUDAExecutionProvider"]
        else:
            model = model.cpu()
            providers = ["CPUExecutionProvider"]

        # Load ONNX model
        logging.info("Loading ONNX model for validation...")
        onnx_model = OnnxWrappedSpanMarkerModel(
            onnx_path=model_onnx_path,
            config=model.config,
            tokenizer=model.tokenizer,
            providers=providers,
        )

        # Obtain predictions for accuracy validation and benchmark
        logging.info("Computing predictions for PyTorch model ...")
        start_ns = time.perf_counter_ns()
        torch_predictions = model.predict(tokens, batch_size=args.batch_size, show_progress_bar=True)
        torch_ms = (time.perf_counter_ns() - start_ns) / 1e6
        avg_torch_ms = torch_ms / len(tokens)
        logging.info(f"Average prediction latency with Torch: {avg_torch_ms:.2f}ms")

        logging.info(f"Computing predictions for ONNX model...")
        start_ns = time.perf_counter_ns()
        onnx_predictions = onnx_model.predict(tokens, batch_size=args.batch_size, show_progress_bar=True)
        onnx_ms = (time.perf_counter_ns() - start_ns) / 1e6
        avg_onnx_ms = onnx_ms / len(tokens)
        logging.info(f"Average prediction latency with ONNX: {avg_onnx_ms:.2f}ms")

        # Check predictions
        logging.info("Comparing predictions...")
        mismatched_predictions = []

        for i, (torch_pred, onnx_pred) in enumerate(zip(torch_predictions, onnx_predictions)):
            torch_ents = list(sorted([(ent["span"], ent["label"]) for ent in torch_pred]))
            onnx_ents = list(sorted([(ent["span"], ent["label"]) for ent in onnx_pred]))

            if torch_ents != onnx_ents:
                mismatched_predictions.append((i, torch_ents, onnx_ents))

        if mismatched_predictions:
            logging.info(f"Found {len(mismatched_predictions)} mismatched predictions between PyTorch and ONNX models:")
            for idx, torch_ents, onnx_ents in mismatched_predictions[:5]:  # Show first 5 mismatches
                logging.info(f"Tokens: {tokens[idx]}")
                logging.info(f"Expected predictions (PyTorch): {torch_ents}")
                logging.info(f"Actual predictions (ONNX): {onnx_ents}")
                logging.info("=" * 20)
            if len(mismatched_predictions) > 5:
                logging.info(f"... and {len(mismatched_predictions) - 5} more mismatches")
        else:
            logging.info("All predictions match between PyTorch and ONNX models! Conversion successful.")

    # Generate and save Triton config.pbtxt
    logging.info(f"Generating Triton repository at {triton_model_dir.parent}")

    # Create Triton model directory structure
    triton_model_version_dir.mkdir(parents=True, exist_ok=True)
    triton_onnx_path.write_bytes(model_onnx_path.read_bytes())

    onnx_proto = onnx.load(triton_onnx_path.as_posix())
    inputs = [io_to_tuple(v) for v in onnx_proto.graph.input]
    outputs = [io_to_tuple(v) for v in onnx_proto.graph.output]

    triton_config = make_triton_config(
        model_name=args.triton_model_name,
        backend=args.triton_backend,
        inputs=inputs,
        outputs=outputs,
        gpu_ids=args.gpu_ids,
    )

    with open(triton_config_path, "w") as f:
        f.write(triton_config)
    logging.info("Triton repository ready!")


if __name__ == "__main__":
    main()
