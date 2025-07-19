import math
from typing import Any, Dict, List, Optional

from datasets import Dataset
from tqdm.auto import tqdm


def spread_sample(batch: Dict[str, List[Any]], model_max_length: int, marker_max_length: int) -> Dict[str, List[Any]]:
    """Spread sentences between multiple samples if lack of space per sample requires it.

    Args:
        batch (`Dict[str, List[Any]]`): A dictionary of dataset keys to lists of values.
        model_max_length (`int`): The total number of tokens that can be processed before
            truncation.
        marker_max_length (`int`): The maximum length for each of the span markers. A value of 128
            means that each training and inferencing sample contains a maximum of 128 start markers
            and 128 end markers, for a total of 256 markers per sample.

    Returns:
        Dict[str, List[Any]]: A dictionary of dataset keys to lists of values.
    """
    keys = batch.keys()
    values = batch.values()
    total_sample_length = model_max_length + 2 * marker_max_length

    batch_samples = {key: [] for key in keys}
    for sample in zip(*values):
        sample = dict(zip(keys, sample))
        sample_marker_space = (total_sample_length - len(sample["input_ids"])) // 2
        spread_between_n = math.ceil(len(sample["start_position_ids"]) / sample_marker_space)
        for i in range(spread_between_n):
            sample_copy = sample.copy()
            start = i * sample_marker_space
            end = (i + 1) * sample_marker_space
            sample_copy["start_position_ids"] = sample["start_position_ids"][start:end]
            sample_copy["end_position_ids"] = sample["end_position_ids"][start:end]
            if "labels" in sample:
                sample_copy["labels"] = sample["labels"][start:end]
            sample_copy["num_spans"] = len(sample_copy["start_position_ids"])
            for key, value in sample_copy.items():
                batch_samples[key].append(value)
    return batch_samples


def add_context(
    dataset: Dataset,
    model_max_length: int,
    max_prev_context: Optional[int] = None,
    max_next_context: Optional[int] = None,
    show_progress_bar: bool = True,
) -> Dataset:
    """Add document-level context from previous and next sentences in the same document.

    Args:
        dataset (`Dataset`): The partially processed dataset, containing `"input_ids"`, `"start_position_ids"`,
            `"end_position_ids"`, `"document_id"` and `"sentence_id"` columns.
        model_max_length (`int`): The total number of tokens that can be processed before
            truncation.
        max_prev_context (`Optional[int]`): The maximum number of previous sentences to include. Defaults to None,
            representing as many previous sentences as fits.
        max_next_context (`Optional[int]`): The maximum number of next sentences to include. Defaults to None,
            representing as many previous sentences as fits.
        show_progress_bar (`bool`): Whether to show a progress bar. Defaults to `True`.

    Returns:
        Dataset: A copy of the Dataset with additional previous and next sentences added to input_ids.
    """
    all_input_ids = []
    all_start_position_ids = []
    all_end_position_ids = []
    for sample_idx, sample in tqdm(
        enumerate(dataset),
        desc="Adding document-level context",
        total=len(dataset),
        leave=False,
        disable=not show_progress_bar,
    ):
        # Sequentially add next context, previous context, next context, previous context, etc. until
        # max token length or max_prev/next_context
        tokens = sample["input_ids"][1:-1]
        start_position_ids = sample["start_position_ids"]
        end_position_ids = sample["end_position_ids"]

        next_context_added = 0
        prev_context_added = 0
        remaining_space = model_max_length - len(tokens) - 2
        while remaining_space > 0:
            next_context_index = sample_idx + next_context_added + 1
            should_add_next = (
                (max_next_context is None or next_context_added < max_next_context)
                and next_context_index < len(dataset)
                and dataset[next_context_index]["document_id"] == sample["document_id"]
            )
            if should_add_next:
                # TODO: [1:-1][:remaining_space] is not efficient
                tokens += dataset[next_context_index]["input_ids"][1:-1][:remaining_space]
                next_context_added += 1

            remaining_space = model_max_length - len(tokens) - 2
            if remaining_space <= 0:
                break

            prev_context_index = sample_idx - prev_context_added - 1
            should_add_prev = (
                (max_prev_context is None or prev_context_added < max_prev_context)
                and prev_context_index >= 0
                and dataset[prev_context_index]["document_id"] == sample["document_id"]
            )
            if should_add_prev:
                # TODO: [1:-1][remaining_space:] is not efficient
                prepended_tokens = dataset[prev_context_index]["input_ids"][1:-1][-remaining_space:]
                tokens = prepended_tokens + tokens
                # TODO: Use numpy? np.array(sample["start_position_ids"]) + len(prepended_tokens)
                start_position_ids = [index + len(prepended_tokens) for index in start_position_ids]
                end_position_ids = [index + len(prepended_tokens) for index in end_position_ids]
                prev_context_added += 1

            if not should_add_next and not should_add_prev:
                break

            remaining_space = model_max_length - len(tokens) - 2

        all_input_ids.append([sample["input_ids"][0]] + tokens + [sample["input_ids"][-1]])
        all_start_position_ids.append(start_position_ids)
        all_end_position_ids.append(end_position_ids)

    dataset = dataset.remove_columns(("input_ids", "start_position_ids", "end_position_ids"))
    dataset = dataset.add_column("input_ids", all_input_ids)
    dataset = dataset.add_column("start_position_ids", all_start_position_ids)
    dataset = dataset.add_column("end_position_ids", all_end_position_ids)

    return dataset
