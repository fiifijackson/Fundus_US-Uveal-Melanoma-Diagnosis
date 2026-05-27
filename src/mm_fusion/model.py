"""Model definitions for the multi-modal CBAM fusion classifier."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import (
    Add,
    Concatenate,
    Conv2D,
    Dense,
    Flatten,
    GlobalAveragePooling2D,
    GlobalMaxPooling2D,
    Input,
    Layer,
    Multiply,
    Reshape,
)
from tensorflow.keras.models import Model, load_model


class TrainableWeightedAverage(Layer):
    """Trainable weighted average layer retained from the original experimental code."""

    def __init__(self, initial_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.initial_weights = initial_weights

    def build(self, input_shape):
        if self.initial_weights is None:
            self.initial_weights = [1.0] * len(input_shape)

        self.kernel = self.add_weight(
            name="kernel",
            shape=(len(input_shape),),
            initializer=tf.keras.initializers.Constant(self.initial_weights),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        clipped_weights = tf.nn.relu(self.kernel)
        weighted_sum = tf.reduce_sum(
            [clipped_weights[i] * inputs[i] for i in range(len(inputs))], axis=0
        )
        return weighted_sum / tf.reduce_sum(clipped_weights)

    def get_config(self):
        config = super().get_config()
        config.update({"initial_weights": self.initial_weights})
        return config


class WeightLogger(Callback):
    """Log changes in a layer's weights after each epoch."""

    def __init__(self, layer_name: str):
        super().__init__()
        self.layer_name = layer_name
        self.previous_weights = None

    def on_train_begin(self, logs=None):
        layer = self.model.get_layer(self.layer_name)
        self.previous_weights = layer.get_weights()

    def on_epoch_end(self, epoch, logs=None):
        layer = self.model.get_layer(self.layer_name)
        current_weights = layer.get_weights()
        for i, (previous, current) in enumerate(zip(self.previous_weights, current_weights)):
            change = np.sum(np.abs(previous - current))
            print(
                f"Epoch {epoch + 1}, weight change in layer "
                f"'{self.layer_name}', weight set {i}: {change}"
            )
        self.previous_weights = current_weights


def channel_attention(input_tensor, reduction_ratio: int = 4):
    """Channel attention module used by CBAM."""
    channels = int(input_tensor.shape[-1])

    avg_pool = GlobalAveragePooling2D()(input_tensor)
    max_pool = GlobalMaxPooling2D()(input_tensor)

    avg_pool = Reshape((1, 1, channels))(avg_pool)
    max_pool = Reshape((1, 1, channels))(max_pool)

    shared_dense = Dense(max(channels // reduction_ratio, 1), activation="relu")
    mlp_out_avg = shared_dense(avg_pool)
    mlp_out_max = shared_dense(max_pool)

    mlp_out_avg = Dense(channels, activation="sigmoid")(mlp_out_avg)
    mlp_out_max = Dense(channels, activation="sigmoid")(mlp_out_max)

    attention_map = Add()([mlp_out_avg, mlp_out_max])
    return Multiply()([input_tensor, attention_map])


def spatial_attention(input_tensor):
    """Spatial attention module used by CBAM."""
    avg_pool = tf.reduce_mean(input_tensor, axis=-1, keepdims=True)
    max_pool = tf.reduce_max(input_tensor, axis=-1, keepdims=True)

    concat = Concatenate(axis=-1)([avg_pool, max_pool])
    attention_map = Conv2D(
        filters=1, kernel_size=7, padding="same", activation="sigmoid"
    )(concat)

    return Multiply()([input_tensor, attention_map])


def cbam_block(input_tensor, reduction_ratio: int = 4):
    """Convolutional Block Attention Module."""
    x = channel_attention(input_tensor, reduction_ratio)
    x = spatial_attention(x)
    return x


def create_frozen_cbam_submodel(
    architecture_path: str | Path,
    weights_path: str | Path,
    name: str,
    feature_layer_name: str = "top_activation",
) -> Model:
    """Load a trained single-modality model, freeze it, and add CBAM to its feature map."""
    base_model = load_model(str(architecture_path))
    base_model.load_weights(str(weights_path))

    for layer in base_model.layers:
        layer.trainable = False

    features = base_model.get_layer(feature_layer_name).output
    features = cbam_block(features)

    return Model(inputs=base_model.input, outputs=features, name=f"{name}_model")


def build_fusion_model(
    input_shape: tuple[int, int, int],
    base_architecture_path: str | Path,
    optos_weights_path: str | Path,
    lus_weights_path: str | Path,
    tus_weights_path: str | Path,
) -> Model:
    """Build the three-input CBAM fusion model."""
    optos_submodel = create_frozen_cbam_submodel(
        base_architecture_path, optos_weights_path, name="optos"
    )
    lus_submodel = create_frozen_cbam_submodel(
        base_architecture_path, lus_weights_path, name="lus"
    )
    tus_submodel = create_frozen_cbam_submodel(
        base_architecture_path, tus_weights_path, name="tus"
    )

    optos_input = Input(shape=input_shape, name="optos_input")
    lus_input = Input(shape=input_shape, name="lus_input")
    tus_input = Input(shape=input_shape, name="tus_input")

    optos_features = optos_submodel(optos_input)
    lus_features = lus_submodel(lus_input)
    tus_features = tus_submodel(tus_input)

    combined_features = Concatenate(name="feature_concatenation")(
        [optos_features, lus_features, tus_features]
    )
    output = Flatten(name="flatten")(combined_features)
    output = Dense(1, activation="sigmoid", name="classification_output")(output)

    return Model(
        inputs=[optos_input, lus_input, tus_input],
        outputs=output,
        name="cbam_multimodal_fusion_model",
    )
