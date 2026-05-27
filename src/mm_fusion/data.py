"""Data loading and generator utilities for multi-modal image fusion."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator


@dataclass(frozen=True)
class DataConfig:
    """Paths and column names used to load the three imaging modalities."""

    optos_dir: Path
    lus_dir: Path
    tus_dir: Path
    optos_csv: Path
    lus_csv: Path
    tus_csv: Path
    filename_col: str = "Image Name"
    label_col: str = "Class"


def load_modality_tables(config: DataConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the CSV files for Optos, LUS, and TUS images."""
    optos_df = pd.read_csv(config.optos_csv)
    lus_df = pd.read_csv(config.lus_csv)
    tus_df = pd.read_csv(config.tus_csv)

    for name, df in {"optos": optos_df, "lus": lus_df, "tus": tus_df}.items():
        missing = {config.filename_col, config.label_col} - set(df.columns)
        if missing:
            raise ValueError(f"{name} CSV is missing required columns: {sorted(missing)}")

    if not (len(optos_df) == len(lus_df) == len(tus_df)):
        raise ValueError("The three modality CSV files must contain the same number of rows.")

    if not (
        optos_df[config.label_col].to_numpy() == lus_df[config.label_col].to_numpy()
    ).all() or not (
        optos_df[config.label_col].to_numpy() == tus_df[config.label_col].to_numpy()
    ).all():
        raise ValueError(
            "Labels do not match across modalities. Ensure rows are aligned across CSV files."
        )

    return optos_df, lus_df, tus_df


def make_image_generators(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    image_dir: Path,
    filename_col: str,
    label_col: str,
    image_size: tuple[int, int],
    batch_size: int,
    class_mode: str,
    train_datagen: ImageDataGenerator,
    val_datagen: ImageDataGenerator,
    seed: int = 42,
):
    """Create matched train and validation generators for a single modality."""
    train_gen = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=str(image_dir),
        x_col=filename_col,
        y_col=label_col,
        seed=seed,
        target_size=image_size,
        class_mode=class_mode,
        color_mode="rgb",
        batch_size=batch_size,
        shuffle=False,
    )

    val_gen = val_datagen.flow_from_dataframe(
        dataframe=val_df,
        directory=str(image_dir),
        x_col=filename_col,
        y_col=label_col,
        seed=seed,
        target_size=image_size,
        class_mode=class_mode,
        color_mode="rgb",
        batch_size=batch_size,
        shuffle=False,
    )

    return train_gen, val_gen


def combined_generator(generator_optos, generator_lus, generator_tus) -> Iterator:
    """Yield synchronized batches from three modality-specific generators."""
    while True:
        optos_batch = next(generator_optos)
        lus_batch = next(generator_lus)
        tus_batch = next(generator_tus)

        if not (
            np.array_equal(optos_batch[1], lus_batch[1])
            and np.array_equal(lus_batch[1], tus_batch[1])
        ):
            raise ValueError("Labels do not match between modality generators.")

        yield [optos_batch[0], lus_batch[0], tus_batch[0]], optos_batch[1]
