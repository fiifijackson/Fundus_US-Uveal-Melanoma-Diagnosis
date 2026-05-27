"""Train and evaluate the multi-modal CBAM fusion model with stratified cross-validation.

Example
-------
python scripts/train_cross_validation.py --config configs/config.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from mm_fusion.data import DataConfig, combined_generator, load_modality_tables, make_image_generators
from mm_fusion.metrics import (
    evaluate_binary_classifier,
    plot_confusion_matrix,
    plot_roc_curve,
    save_prediction_table,
)
from mm_fusion.model import build_fusion_model
from mm_fusion.utils import configure_gpu_memory_growth, ensure_dir, load_config, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train multi-modal CBAM fusion model.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/config.yaml"),
        help="Path to YAML config file.",
    )
    return parser.parse_args()


def make_fold_dataframe(source_df: pd.DataFrame, train_index, val_index, filename_col, label_col):
    train_df = pd.DataFrame(
        {
            filename_col: source_df[filename_col].iloc[train_index].values,
            label_col: source_df[label_col].iloc[train_index].values,
        }
    )
    val_df = pd.DataFrame(
        {
            filename_col: source_df[filename_col].iloc[val_index].values,
            label_col: source_df[label_col].iloc[val_index].values,
        }
    )
    return train_df, val_df


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    set_seed(cfg["seed"])
    configure_gpu_memory_growth()
    output_dir = ensure_dir(cfg["outputs"]["dir"])

    filename_col = cfg["columns"]["filename"]
    label_col = cfg["columns"]["label"]
    image_size = (cfg["image"]["height"], cfg["image"]["width"])
    input_shape = (cfg["image"]["height"], cfg["image"]["width"], cfg["image"]["channels"])

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

    class_mode = "binary" if len(cfg["class_names"]) <= 2 else "categorical"
    loss_function = "binary_crossentropy" if class_mode == "binary" else "categorical_crossentropy"

    train_datagen = ImageDataGenerator(**cfg["augmentation"])
    val_datagen = ImageDataGenerator(**cfg["validation_preprocessing"])

    metrics = [
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.AUC(name="auc"),
    ]

    kfold = StratifiedKFold(
        n_splits=cfg["num_folds"],
        shuffle=True,
        random_state=cfg["seed"],
    )

    fold_metrics = []
    labels = optos_df[label_col]

    for fold, (train_index, val_index) in enumerate(kfold.split(optos_df[filename_col], labels), start=1):
        print(f"\n========== Fold {fold}/{cfg['num_folds']} ==========")

        optos_train_df, optos_val_df = make_fold_dataframe(
            optos_df, train_index, val_index, filename_col, label_col
        )
        lus_train_df, lus_val_df = make_fold_dataframe(
            lus_df, train_index, val_index, filename_col, label_col
        )
        tus_train_df, tus_val_df = make_fold_dataframe(
            tus_df, train_index, val_index, filename_col, label_col
        )

        optos_train_gen, optos_val_gen = make_image_generators(
            optos_train_df,
            optos_val_df,
            data_cfg.optos_dir,
            filename_col,
            label_col,
            image_size,
            cfg["batch_size"],
            class_mode,
            train_datagen,
            val_datagen,
            seed=cfg["seed"],
        )
        lus_train_gen, lus_val_gen = make_image_generators(
            lus_train_df,
            lus_val_df,
            data_cfg.lus_dir,
            filename_col,
            label_col,
            image_size,
            cfg["batch_size"],
            class_mode,
            train_datagen,
            val_datagen,
            seed=cfg["seed"],
        )
        tus_train_gen, tus_val_gen = make_image_generators(
            tus_train_df,
            tus_val_df,
            data_cfg.tus_dir,
            filename_col,
            label_col,
            image_size,
            cfg["batch_size"],
            class_mode,
            train_datagen,
            val_datagen,
            seed=cfg["seed"],
        )

        model = build_fusion_model(
            input_shape=input_shape,
            base_architecture_path=cfg["model"]["base_architecture_path"],
            optos_weights_path=cfg["model"]["optos_weights_pattern"].format(fold=fold),
            lus_weights_path=cfg["model"]["lus_weights_pattern"].format(fold=fold),
            tus_weights_path=cfg["model"]["tus_weights_pattern"].format(fold=fold),
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=cfg["learning_rate"]),
            loss=loss_function,
            metrics=metrics,
        )

        checkpoint_path = cfg["model"]["checkpoint_pattern"].format(fold=fold)
        callbacks = [
            ModelCheckpoint(
                checkpoint_path,
                save_best_only=True,
                save_weights_only=True,
                monitor="val_accuracy",
                mode="max",
                verbose=2,
            ),
            EarlyStopping(
                monitor="val_accuracy",
                mode="max",
                patience=20,
                restore_best_weights=True,
                verbose=1,
            ),
            ReduceLROnPlateau(
                monitor="val_accuracy",
                factor=0.1,
                patience=20,
                min_lr=1e-6,
                verbose=1,
            ),
        ]

        train_gen = combined_generator(optos_train_gen, lus_train_gen, tus_train_gen)
        val_gen = combined_generator(optos_val_gen, lus_val_gen, tus_val_gen)

        model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=cfg["epochs"],
            steps_per_epoch=len(optos_train_gen),
            validation_steps=len(optos_val_gen),
            callbacks=callbacks,
            class_weight={int(k): float(v) for k, v in cfg["class_weights"].items()},
        )

        # Re-create validation generator before prediction to ensure it starts at the first sample.
        _, optos_val_gen = make_image_generators(
            optos_train_df,
            optos_val_df,
            data_cfg.optos_dir,
            filename_col,
            label_col,
            image_size,
            cfg["batch_size"],
            class_mode,
            train_datagen,
            val_datagen,
            seed=cfg["seed"],
        )
        _, lus_val_gen = make_image_generators(
            lus_train_df,
            lus_val_df,
            data_cfg.lus_dir,
            filename_col,
            label_col,
            image_size,
            cfg["batch_size"],
            class_mode,
            train_datagen,
            val_datagen,
            seed=cfg["seed"],
        )
        _, tus_val_gen = make_image_generators(
            tus_train_df,
            tus_val_df,
            data_cfg.tus_dir,
            filename_col,
            label_col,
            image_size,
            cfg["batch_size"],
            class_mode,
            train_datagen,
            val_datagen,
            seed=cfg["seed"],
        )

        val_gen = combined_generator(optos_val_gen, lus_val_gen, tus_val_gen)
        y_pred_probs = model.predict(val_gen, steps=len(optos_val_gen), verbose=1)
        y_true = optos_val_gen.classes

        results = evaluate_binary_classifier(y_true, y_pred_probs)
        fold_metrics.append(
            {
                "Fold": fold,
                "Accuracy": results["accuracy"],
                "Sensitivity": results["sensitivity"],
                "Specificity": results["specificity"],
                "Precision": results["precision"],
                "F1-Score": results["f1_score"],
                "AUC": results["auc"],
            }
        )

        save_prediction_table(
            optos_val_gen.filenames,
            y_true,
            results["y_pred"],
            output_dir / f"prediction_results_fold_{fold}.xlsx",
        )
        plot_confusion_matrix(
            results["confusion_matrix"],
            cfg["class_names"],
            output_dir / f"confusion_matrix_fold_{fold}.png",
        )
        plot_roc_curve(
            y_true,
            y_pred_probs,
            output_dir / f"roc_curve_fold_{fold}.png",
        )

        print(
            f"Fold {fold}: "
            f"accuracy={results['accuracy']:.4f}, "
            f"sensitivity={results['sensitivity']:.4f}, "
            f"specificity={results['specificity']:.4f}, "
            f"precision={results['precision']:.4f}, "
            f"f1={results['f1_score']:.4f}, "
            f"auc={results['auc']:.4f}"
        )

    metrics_df = pd.DataFrame(fold_metrics)
    metrics_df.to_excel(cfg["outputs"]["metrics_file"], index=False)
    print(f"\nSaved cross-validation metrics to {cfg['outputs']['metrics_file']}")


if __name__ == "__main__":
    main()
