from __future__ import annotations

from tensorflow import keras


def build_dqn_model(number_of_actions: int = 40) -> keras.Model:
    board_input = keras.layers.Input(shape=(20, 10, 1), name="board_input")
    current_piece_input = keras.layers.Input(shape=(7,), name="current_piece_input")
    next_piece_input = keras.layers.Input(shape=(7,), name="next_piece_input")

    board_features = keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same")(board_input)
    board_features = keras.layers.Flatten()(board_features)

    piece_features = keras.layers.Concatenate()([current_piece_input, next_piece_input])

    features = keras.layers.Concatenate()([board_features, piece_features])
    features = keras.layers.Dense(128, activation="relu")(features)
    q_values = keras.layers.Dense(number_of_actions, activation="linear")(features)

    model = keras.Model(inputs=[board_input, current_piece_input, next_piece_input], outputs=q_values)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mse")
    return model
