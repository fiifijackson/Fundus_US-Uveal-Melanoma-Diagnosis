"""General project utilities."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import tensorflow as tf
import yaml


def load_config(path: str | Path) -> dict:
    """Load a YAML configuration file."""
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def configure_gpu_memory_growth() -> None:
    """Enable TensorFlow GPU memory growth when GPUs are available."""
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"Num GPUs Available: {len(gpus)}")


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not already exist and return it as a Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
