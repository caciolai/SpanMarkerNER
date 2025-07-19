import os
import random
from pathlib import Path

import hydra
import numpy as np
import torch
from datasets import load_dataset
from omegaconf import DictConfig
from transformers import Trainer

from span_marker import SpanMarkerModel

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@hydra.main(config_path="hydra_configs", config_name="baseline", version_base=None)
def main(cfg: DictConfig) -> None:
    # Seed everything
    seed = cfg.experiment.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Load the dataset, ensure "tokens" and "ner_tags" columns, and get a list of labels
    dataset = load_dataset(cfg.dataset.id, trust_remote_code=True)
    labels = dataset["train"].features["ner_tags"].feature.names

    # Instantiate model
    model: SpanMarkerModel = hydra.utils.instantiate(cfg.model, labels=labels)

    # Prepare the 🤗 transformers training arguments

    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
    )
    trainer.train()
    checkpoint_path = Path(cfg.experiment.checkpoint_path)
    checkpoint_path.parent.mkdir(exist_ok=True, parents=True)
    trainer.save_model(checkpoint_path)

    # Compute & save the metrics on the test set
    metrics = trainer.evaluate(dataset["test"], metric_key_prefix="test")
    trainer.save_metrics("test", metrics)


if __name__ == "__main__":
    main()
