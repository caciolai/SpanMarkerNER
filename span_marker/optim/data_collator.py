from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from torch.nn import functional as F

from span_marker.tokenizer import SpanMarkerTokenizer


@dataclass
class SpanMarkerDataCollator:
    """
    Data Collator class responsible for converting the minimal outputs from the tokenizer into
    complete and meaningful inputs to the model. In particular, the ``input_ids`` from the tokenizer
    features are padded, and the correct amount of start and end markers (with padding) are added.

    Furthermore, the position IDs are generated for the input IDs, and ``start_position_ids`` and
    ``end_position_ids`` are used alongside some padding to create a full position ID vector.

    Lastly, the attention matrix is computed.

    The expected usage is something like:

    >>> collator = SpanMarkerDataCollator(...)
    >>> tokenized = tokenizer(...)
    >>> batch = collator(tokenized)
    >>> output = model(**batch)
    """

    tokenizer: SpanMarkerTokenizer
    marker_max_length: int

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Convert the minimal tokenizer outputs into inputs ready for :meth:`~span_marker.modeling.SpanMarkerModel.forward`.

        Args:
            features (List[Dict[str, Any]]): A list of dictionaries, one element per sample in the batch.
                The dictionaries contain the following keys:

                * ``input_ids``: The non-padded input IDs.
                * ``num_spans``: The number of spans that should be encoded in each sample.
                * ``start_position_ids``: The position IDs of the start markers in the sample.
                * ``end_position_ids``: The position IDs of the end markers in the sample.
                * ``labels`` (optional): The labels corresponding to each of the spans in the sample.
                * ``num_words`` (optional): The number of words in the input sample.
                    Required for some evaluation metrics.

        Returns:
            Dict[str, torch.Tensor]: Batch dictionary ready to be fed into :meth:`~span_marker.modeling.SpanMarkerModel.forward`.
        """

        all_input_ids = []
        all_position_ids = []
        all_attention_mask = []
        all_labels = []
        all_num_words = []
        all_num_marker_pairs = []
        all_document_ids = []
        all_sentence_ids = []

        for sample in features:

            input_ids = self._prepare_input_ids(sample)
            position_ids = self._prepare_position_ids(sample)
            attention_mask = self._prepare_attention_mask(sample)
            labels_or_none = self._prepare_labels(sample)

            all_input_ids.append(input_ids)
            all_position_ids.append(position_ids)
            all_attention_mask.append(attention_mask)

            if labels_or_none is not None:
                all_labels.append(labels_or_none)

            if "num_spans" in sample:
                all_num_marker_pairs.append(sample["num_spans"])

            if "num_words" in sample:
                all_num_words.append(sample["num_words"])

            if "document_id" in sample:
                all_document_ids.append(sample["document_id"])

            if "sentence_id" in sample:
                all_sentence_ids.append(sample["sentence_id"])

        batch = dict()
        batch["input_ids"] = torch.stack(tensors=all_input_ids, dim=0)
        batch["position_ids"] = torch.stack(tensors=all_position_ids, dim=0)
        batch["attention_mask"] = torch.stack(tensors=all_attention_mask, dim=0)

        # Used for training
        if all_labels:
            batch["labels"] = torch.stack(tensors=all_labels, dim=0)

        # Used for evaluation / prediction, does not need to be padded/stacked
        if all_num_marker_pairs:
            batch["num_marker_pairs"] = torch.tensor(all_num_marker_pairs, dtype=torch.int)

        if all_num_words:
            batch["num_words"] = torch.tensor(all_num_words, dtype=torch.int)

        if all_document_ids:
            batch["document_ids"] = torch.tensor(all_document_ids, dtype=torch.int)

        if all_sentence_ids:
            batch["sentence_ids"] = torch.tensor(all_sentence_ids, dtype=torch.int)

        return batch

    def _prepare_input_ids(self, sample: Dict[str, Any]) -> torch.Tensor:

        # --- Extract sizes
        input_ids = sample["input_ids"]
        num_spans = sample["num_spans"]
        num_tokens = len(input_ids)

        token_size = self.tokenizer.model_max_length
        marker_size = self.marker_max_length
        total_size = token_size + 2 * marker_size

        start_markers_initial_idx = token_size
        end_markers_initial_idx = start_markers_initial_idx + marker_size

        # --- Prepare text tokens
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids, dtype=torch.int)
        else:
            input_ids.to(torch.int)

        # --- Prepare marker tokens
        input_ids = F.pad(input_ids, (0, total_size - num_tokens), value=self.tokenizer.pad_token_id)
        input_ids[start_markers_initial_idx : start_markers_initial_idx + num_spans] = self.tokenizer.start_marker_id
        input_ids[end_markers_initial_idx : end_markers_initial_idx + num_spans] = self.tokenizer.end_marker_id

        return input_ids

    def _prepare_position_ids(self, sample: Dict[str, Any]) -> torch.Tensor:
        """Prepare position IDs tensor for SpanMarker

        Args:
            sample (Dict[str, Any]): sample with features

        Returns:
            torch.Tensor: position IDs tensor
        """

        # --- Extract sizes
        input_ids = sample["input_ids"]
        num_spans = sample["num_spans"]
        num_tokens = len(input_ids)

        token_size = self.tokenizer.model_max_length
        marker_size = self.marker_max_length
        total_size = token_size + 2 * marker_size

        start_markers_initial_idx = token_size
        end_markers_initial_idx = start_markers_initial_idx + marker_size

        # --- Prepare position IDs
        # Increase the position_ids by 2, inspired by PL-Marker. The intuition is that these position IDs
        # better match the circumstances under which the underlying encoders are trained.
        position_ids = torch.arange(num_tokens, dtype=torch.int) + 2
        position_ids = F.pad(position_ids, (0, total_size - len(position_ids)), value=1)

        # Let start marker tokens refer to the starting token of their span
        position_ids[start_markers_initial_idx : start_markers_initial_idx + num_spans] = (
            torch.tensor(sample["start_position_ids"]) + 2
        )

        # Let end marker tokens refer to the starting token of their span
        position_ids[end_markers_initial_idx : end_markers_initial_idx + num_spans] = (
            torch.tensor(sample["end_position_ids"]) + 2
        )

        return position_ids

    def _prepare_attention_mask_simple(self, sample: Dict[str, Any]) -> torch.Tensor:
        """Prepare attention mask tensor for SpanMarker

        Args:
            sample (Dict[str, Any]): sample with features

        Returns:
            torch.Tensor: attention mark tensor
        """

        # --- Extract sizes
        input_ids = sample["input_ids"]
        num_spans = sample["num_spans"]
        num_tokens = len(input_ids)

        token_size = self.tokenizer.model_max_length
        marker_size = self.marker_max_length
        total_size = token_size + 2 * marker_size

        start_markers_initial_idx = token_size
        end_markers_initial_idx = start_markers_initial_idx + marker_size

        # --- Prepare attention mask matrix
        attention_mask = torch.zeros((total_size), dtype=torch.bool)

        # --- Text tokens (self-attention)
        attention_mask[:num_tokens] = 1

        # --- Marker tokens
        start_markers_indices = list(range(start_markers_initial_idx, start_markers_initial_idx + num_spans))
        end_markers_indices = list(range(end_markers_initial_idx, end_markers_initial_idx + num_spans))

        # Start marker tokens
        attention_mask[start_markers_indices] = 1

        # End marker tokens
        attention_mask[end_markers_indices] = 1

        return attention_mask

    def _prepare_attention_mask(self, sample: Dict[str, Any]) -> torch.Tensor:
        """Prepare attention mask tensor for SpanMarker

        Args:
            sample (Dict[str, Any]): sample with features

        Returns:
            torch.Tensor: attention mark tensor
        """

        # --- Extract sizes
        input_ids = sample["input_ids"]
        num_spans = sample["num_spans"]
        num_tokens = len(input_ids)

        token_size = self.tokenizer.model_max_length
        marker_size = self.marker_max_length
        total_size = token_size + 2 * marker_size

        start_markers_initial_idx = token_size
        end_markers_initial_idx = start_markers_initial_idx + marker_size

        # --- Prepare attention mask matrix
        attention_mask = torch.zeros((total_size, total_size), dtype=torch.bool)

        # --- Text tokens (self-attention)
        attention_mask[:num_tokens, :num_tokens] = 1

        # --- Marker tokens
        start_markers_indices = list(range(start_markers_initial_idx, start_markers_initial_idx + num_spans))
        end_markers_indices = list(range(end_markers_initial_idx, end_markers_initial_idx + num_spans))

        # Start marker tokens can attend to text tokens
        attention_mask[start_markers_initial_idx : start_markers_initial_idx + num_spans, :num_tokens] = 1

        # End marker tokens can attend to text tokens
        attention_mask[end_markers_initial_idx : end_markers_initial_idx + num_spans, :num_tokens] = 1

        # Start marker tokens can attend to themselves
        attention_mask[start_markers_indices, start_markers_indices] = 1

        # Start marker tokens can attend to end marker tokens
        attention_mask[start_markers_indices, end_markers_indices] = 1

        # End marker tokens can attend to themselves
        attention_mask[end_markers_indices, end_markers_indices] = 1

        # End marker tokens can attend to start marker tokens
        attention_mask[end_markers_indices, start_markers_indices] = 1

        return attention_mask

    def _prepare_labels(self, sample: Dict[str, Any]) -> Optional[torch.Tensor]:
        """Prepare labels for SpanMarker, if present in the sample

        Args:
            sample (Dict[str, Any]): sample with features

        Returns:
            Optional[torch.Tensor]: labels tensor if present
        """
        if "labels" in sample:
            marker_size = self.marker_max_length
            labels = torch.tensor(sample["labels"])
            labels = F.pad(labels, (0, (marker_size) - len(labels)), value=-100)

            return labels

        return None
