from __future__ import annotations

from tensorflow import keras


def build_value_model(input_dim: int = 224) -> keras.Model:
    # Build a simple MLP that predicts value for a candidate board state.
    inputs = keras.layers.Input(shape=(input_dim,), name="state_features")
    x = keras.layers.Dense(256, activation="relu")(inputs)
    x = keras.layers.Dense(128, activation="relu")(x)
    outputs = keras.layers.Dense(1, activation="linear")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mse")
    return model
