from __future__ import annotations

from typing import Any

import numpy as np


def piece_to_one_hot(piece_id: int) -> np.ndarray:
    one_hot = np.zeros(7, dtype=np.float32)
    if 0 <= int(piece_id) < 7:
        one_hot[int(piece_id)] = 1.0
    return one_hot


def make_feature_vector(simulated_state: dict[str, Any], feature_dict: dict[str, float]) -> np.ndarray:
    board = np.asarray(simulated_state["board"], dtype=np.float32).reshape(-1)
    current_piece = piece_to_one_hot(int(simulated_state["current_piece"]))
    next_piece = piece_to_one_hot(int(simulated_state["next_piece"]))
    scalar_features = np.array(
        [
            feature_dict["max_height"],
            feature_dict["aggregate_height"],
            feature_dict["holes"],
            feature_dict["bumpiness"],
            feature_dict["completed_lines"],
            feature_dict["well_sum"],
        ],
        dtype=np.float32,
    )
    return np.concatenate([board, current_piece, next_piece, scalar_features]).astype(np.float32)
