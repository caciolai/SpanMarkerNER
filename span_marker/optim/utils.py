import math
from typing import Any, Dict, List


def spread_sample(batch: Dict[str, List[Any]], model_max_length: int, marker_max_length: int) -> Dict[str, List[Any]]:
    """Spread sentences between multiple samples if lack of space per sample requires it.

    Args:
        batch (`Dict[str, List[Any]]`): A dictionary of dataset keys to lists of values.
        model_max_length (`int`): Not actually used
        marker_max_length (`int`): The maximum length for each of the span markers. A value of 128
            means that each training and inferencing sample contains a maximum of 128 start markers
            and 128 end markers, for a total of 256 markers per sample.

    Returns:
        Dict[str, List[Any]]: A dictionary of dataset keys to lists of values.
    """
    keys = batch.keys()
    values = batch.values()

    batch_samples = {key: [] for key in keys}
    for sample in zip(*values):
        sample = dict(zip(keys, sample))
        spread_between_n = math.ceil(len(sample["start_position_ids"]) / marker_max_length)
        for i in range(spread_between_n):
            sample_copy = sample.copy()
            start = i * marker_max_length
            end = (i + 1) * marker_max_length
            sample_copy["start_position_ids"] = sample["start_position_ids"][start:end]
            sample_copy["end_position_ids"] = sample["end_position_ids"][start:end]
            if "labels" in sample:
                sample_copy["labels"] = sample["labels"][start:end]
            sample_copy["num_spans"] = len(sample_copy["start_position_ids"])
            for key, value in sample_copy.items():
                batch_samples[key].append(value)
    return batch_samples
