"""Run inference with a trained multi-modal fusion model.

This script expects three aligned folders/CSV files, matching the training setup.
Update paths in `configs/config.yaml` before running.

Example
-------
python scripts/predict.py --config configs/config.yaml --model outputs/CBAM_model.h5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from mm_fusion.data import DataConfig, combined_generator, load_modality_tables, make_image_generators
from mm_fusion.model import TrainableWeightedAverage
from mm_fusion.utils import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict with trained multi-modal fusion model.")
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    parser.add_argument("--model", type=Path, required=True, help="Path to trained .h5/.keras model.")
    parser.add_argument("--output", type=Path, default=Path("outputs/predictions.xlsx"))
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    filename_col = cfg["columns"]["filename"]
    label_col = cfg["columns"]["label"]
    image_size = (cfg["image"]["height"], cfg["image"]["width"])
    class_mode = "binary" if len(cfg["class_names"]) <= 2 else "categorical"

    data_cfg = DataConfig(
        optos_dir=Path(cfg["data"]["optos_dir"]),
        lus_dir=Path(cfg["data"]["lus_dir"]),
        tus_dir=Path(cfg["data"]["tus_dir"]),
        optos_csv=Path(cfg["data"]["optos_csv"]),
        lus_csv=Path(cfg["data"]["lus_csv"]),
        tus_csv=Path(cfg["data"]["tus_csv"]),
        filename_col=filename_col,
        label_col=label_col,
    )
    optos_df, lus_df, tus_df = load_modality_tables(data_cfg)

    datagen = ImageDataGenerator(**cfg["validation_preprocessing"])

    # The helper expects train/validation dataframes; for inference, pass the same
    # dataframe as both arguments and use the validation generator.
    _, optos_gen = make_image_generators(
        optos_df, optos_df, data_cfg.optos_dir, filename_col, label_col,
        image_size, cfg["batch_size"], class_mode, datagen, datagen, seed=cfg["seed"]
    )
    _, lus_gen = make_image_generators(
        lus_df, lus_df, data_cfg.lus_dir, filename_col, label_col,
        image_size, cfg["batch_size"], class_mode, datagen, datagen, seed=cfg["seed"]
    )
    _, tus_gen = make_image_generators(
        tus_df, tus_df, data_cfg.tus_dir, filename_col, label_col,
        image_size, cfg["batch_size"], class_mode, datagen, datagen, seed=cfg["seed"]
    )

    model = tf.keras.models.load_model(
        args.model,
        custom_objects={"TrainableWeightedAverage": TrainableWeightedAverage},
    )
    pred_probs = np.ravel(
        model.predict(combined_generator(optos_gen, lus_gen, tus_gen), steps=len(optos_gen))
    )
    pred_labels = (pred_probs >= args.threshold).astype(int)

    output_df = pd.DataFrame(
        {
            "Image Name": optos_gen.filenames,
            "Predicted Probability": pred_probs,
            "Predicted Class Index": pred_labels,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_excel(args.output, index=False)
    print(f"Saved predictions to {args.output}")


if __name__ == "__main__":
    main()
