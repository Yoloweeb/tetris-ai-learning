## Dataset Format

Each sample consists of:

- board: numpy array (20, 10), int8
- current_piece: numpy array (7,), one-hot
- next_piece: numpy array (7,), one-hot (optional)
- action: int (macro placement action index)

Stored as .npz files with keys:
- X_board
- X_current
- X_next
- y_action