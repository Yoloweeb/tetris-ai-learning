from __future__ import annotations

from tensorflow import keras


def build_dqn_model(number_of_actions: int = 40) -> keras.Model:
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(20, 10, 1)),
            keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same"),
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dense(number_of_actions, activation="linear"),
        ]
    )
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mse")
    return model
