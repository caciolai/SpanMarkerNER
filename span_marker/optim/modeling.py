import logging
from typing import Optional

import torch
from torch import Tensor

from span_marker.modeling import SpanMarkerModel as BaseSpanMarkerModel
from span_marker.optim.data_collator import SpanMarkerDataCollator
from span_marker.output import SpanMarkerOutput
from span_marker.tokenizer import SpanMarkerTokenizer

logger = logging.getLogger(__name__)


class SpanMarkerModel(BaseSpanMarkerModel):
    def set_tokenizer(self, tokenizer: SpanMarkerTokenizer) -> None:
        self.tokenizer = tokenizer
        self.data_collator = SpanMarkerDataCollator(
            tokenizer=tokenizer, marker_max_length=self.config.marker_max_length
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        num_marker_pairs: torch.Tensor,
        num_words: Optional[torch.Tensor] = None,
        document_ids: Optional[torch.Tensor] = None,
        sentence_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> SpanMarkerOutput:
        """Forward call of the SpanMarkerModel.

        Args:
            input_ids (~torch.Tensor): Input IDs including start/end markers.
            attention_mask (~torch.Tensor): Attention mask matrix including one-directional attention for markers.
            position_ids (~torch.Tensor): Position IDs including start/end markers.
            num_marker_pairs (~torch.Tensor): The number of start/end marker pairs per batch sample.
            labels (Optional[~torch.Tensor]): The labels for each span candidate. Defaults to None.
            num_words (Optional[~torch.Tensor]): The number of words for each batch sample. Defaults to None.
            document_ids (Optional[~torch.Tensor]): The document ID of each batch sample. Defaults to None.
            sentence_ids (Optional[~torch.Tensor]): The index of each sentence in their respective document. Defaults to None.

        Returns:
            SpanMarkerOutput: The output dataclass.
        """

        # --- Extract sizes
        batch_size = input_ids.size(0)

        token_size = self.config.model_max_length
        marker_size = self.config.marker_max_length

        start_markers_initial_idx = token_size
        end_markers_initial_idx = start_markers_initial_idx + marker_size

        # --- Pass features through encoder
        token_type_ids = torch.zeros_like(input_ids)
        outputs = self.encoder(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )

        # (batch_size, sequence_length, hidden_size)
        last_hidden_state: Tensor = outputs[0]
        last_hidden_state = self.dropout(last_hidden_state)

        # --- Feed marker embeddings through classifier

        # (batch_size, marker_size, hidden_size)
        start_marker_embeddings = last_hidden_state[:, start_markers_initial_idx:end_markers_initial_idx, :]
        end_marker_embeddings = last_hidden_state[:, end_markers_initial_idx:, :]

        # (batch_size, marker_size, 2*hidden_size)
        marker_embeddings = torch.cat([start_marker_embeddings, end_marker_embeddings], dim=-1)
        marker_embeddings = self.dropout(marker_embeddings)

        # (batch_size, marker_size, num_classes)
        logits: Tensor = self.classifier(marker_embeddings)

        # --- Feed marker embeddings through classifier
        if labels is not None:

            # (batch_size * marker_size, num_classes)
            logits_flattened = logits.reshape(batch_size * marker_size, -1)

            # (batch_size * marker_size, )
            labels_flattened = labels.reshape(batch_size * marker_size)

            # ()
            loss = self.loss_func(logits_flattened, labels_flattened)

        return SpanMarkerOutput(
            loss=loss if labels is not None else None,
            logits=logits,
            *outputs[2:],
            num_marker_pairs=num_marker_pairs,
            num_words=num_words,
            document_ids=document_ids,
            sentence_ids=sentence_ids,
        )
