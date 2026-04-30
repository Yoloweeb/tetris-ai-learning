# Tetris AI Project

A compact, presentation-friendly Tetris AI project:
- **NumPy-based environment** (20x10 board)
- AI **simulates valid placements**
- A value model scores resulting board states and selects the best move

## Install

```bash
pip install -r requirements.txt
```

## Train

```bash
python -m src.training.train_dqn
```

## Watch (visual)

```bash
python -m src.training.watch_ai
```

Optional recording-friendly flags:

```bash
python -m src.training.watch_ai --fps 5 --episodes 1 --max-steps 300 --seed 123
```

## Play (terminal)

```bash
python -m src.training.play_dqn
```
