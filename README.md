# Tetris AI Learning

This project is a reinforcement-learning based Tetris AI prototype.

Instead of directly learning fixed button presses, the AI evaluates candidate piece placements, predicts the quality of the resulting board states, and then chooses the best move.

## Python Version

Recommended Python version:

**Python 3.10**

This project was tested with Python 3.10. TensorFlow can be sensitive to very new Python versions, so it is recommended to avoid Python 3.13 or 3.14 for now.

## Installation

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Activate it on Git Bash:

```bash
source .venv/Scripts/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Main Dependencies

- **NumPy**: board simulation and numerical operations
- **TensorFlow / Keras**: neural network training
- **Pygame**: visual AI playback
- **Pytest**: basic testing

## How the AI Works

The current approach is placement evaluation:

1. The environment lists all valid placements.
2. Each placement is simulated.
3. Features are extracted from the resulting board.
4. A neural network scores each resulting board state.
5. The AI chooses the placement with the highest predicted value.

This is more suitable for Tetris than directly predicting one raw action.

## Features Used by the AI

- flattened board grid
- current piece
- next piece
- max height
- aggregate height
- holes
- bumpiness
- completed lines
- well sum
- row transitions
- column transitions
- landing height
- eroded piece cells

## Training

Run training with:

```bash
python -m src.training.train_dqn
```

Training saves models locally inside `models/`.

Important model files:

- `models/tetris_value_latest.keras`
- `models/tetris_value_best.keras`
- `models/tetris_value_best_lines.keras`
- `models/tetris_value_final.keras`

## Evaluation

Run evaluation with:

```bash
python -m src.training.evaluate_dqn
```

It reports:

- average reward
- average lines cleared
- average steps survived
- maximum lines cleared
- percentage of episodes with at least one line clear
- percentage of episodes with at least two line clears

## Visual Playback

Run visual playback with:

```bash
python -m src.training.watch_ai --fps 5 --episodes 1 --max-steps 300
```

This opens a Pygame window and is useful for demo video recording.

## Terminal Playback

Run terminal playback with:

```bash
python -m src.training.play_dqn
```

## Tests

Run tests with:

```bash
python -m pytest
```

## Project Structure

```text
src/
  env/
    tetris_env.py          # NumPy Tetris environment
  training/
    features.py            # Feature extraction for board evaluation
    model.py               # Neural network model
    train_dqn.py           # Training loop
    evaluate_dqn.py        # Evaluation script
    play_dqn.py            # Terminal playback
    watch_ai.py            # Visual playback with Pygame
tests/
  test_env_smoke.py
models/
  trained model checkpoints
```

## Current Limitations

- AI is not optimal
- environment is simplified compared to official Tetris
- model uses handcrafted features
- AI has limited lookahead
- performance depends on training time and reward design

## Future Improvements

- train for more episodes
- tune reward weights
- improve lookahead
- add better board evaluation features
- experiment with larger neural network
- compare against a heuristic Tetris bot

## Main Takeaway

The most important design decision was switching from direct action prediction to placement/state evaluation.

Instead of asking **“which action should I take?”**, the AI asks:

**“Which possible placement leads to the best board state?”**

This makes the learning problem much more suitable for Tetris.

## Demo

https://github.com/user-attachments/assets/02fb39a1-f426-4460-aece-1358680bb6ea



