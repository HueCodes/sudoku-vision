# Sudoku Vision

Scan sudoku puzzles with your camera and solve them instantly using computer vision and machine learning.

## Project Structure

```
sudoku-vision/
├── solver/     # C sudoku solver (compiles to WASM)
├── ml/         # Python ML training pipeline
├── web/        # TypeScript frontend app
├── data/       # Training images and test fixtures
└── ROADMAP.md  # Development plan and architecture
```

## Status

Early development. See [ROADMAP.md](./ROADMAP.md) for the full plan.

## Components

- **Solver:** Backtracking algorithm in C, targeting WebAssembly
- **ML Pipeline:** CNN for digit recognition (PyTorch)
- **Grid Detection:** OpenCV-based cell extraction
- **Web App:** Camera capture, inference, solution display

## License

MIT
