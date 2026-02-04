Define one sample as:

    board: shape (20, 10) int8 (0/1)

    piece_current: shape (7,) one-hot

    piece_next: shape (7,) one-hot (optional)

    action: int (index into “placement actions”)
    Store as .npz with arrays: X_board, X_cur, X_next, y_action.