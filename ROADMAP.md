# Sudoku Scanner + Solver App Roadmap

This document outlines the evolution of C-SudokuSolver from a CLI tool into a full mobile/web app with camera scanning and AI-powered digit recognition.

---

## Current State

The C-SudokuSolver is a clean, ~250 line CLI app using backtracking:
- Solid core algorithm (works for most puzzles)
- Manual input only (no file I/O, no image input)
- No tests, no optimization for hard puzzles
- Good documentation

---

## Vision

**End Goal:** Point your phone camera at a sudoku puzzle, app recognizes it, solves it instantly, overlays the solution.

This requires four major technical components:

```
Camera Input -> Grid Detection -> Digit Recognition -> Solver -> Display Solution
               (Computer Vision)    (ML/CNN)          (Algorithm)
```

---

## Product Architecture Options

### Option A: Mobile-First Native App
- **Platforms:** iOS (Swift) + Android (Kotlin) or cross-platform (Flutter/React Native)
- **ML Runtime:** Core ML (iOS), TensorFlow Lite (both), or ONNX Runtime
- **Pros:** Best camera integration, offline-first, native performance
- **Cons:** Two codebases (unless cross-platform), app store approval process

### Option B: Web App with PWA
- **Stack:** Web (TypeScript) + WebAssembly (for solver) + TensorFlow.js
- **Pros:** Single codebase, instant distribution, no app store
- **Cons:** Camera APIs less mature, performance overhead, offline more complex

### Option C: Hybrid - Web Core + Native Wrapper
- **Stack:** Core logic in Rust/C compiled to WASM, wrapped with Capacitor/Tauri for mobile
- **Pros:** Reuse C solver, single core logic, native distribution
- **Cons:** Complexity in bridging, camera handling still needs native code

**Recommendation:** Start with Option B (Web) for rapid iteration, then wrap for mobile later. Port C solver to WASM to keep the fast core.

---

## Technical Deep Dive: The ML Pipeline

### Phase 1: Grid Detection (Computer Vision)

This is classical CV, not deep learning:

1. **Preprocessing:**
   - Grayscale conversion
   - Gaussian blur to reduce noise
   - Adaptive thresholding (handles varying lighting)

2. **Grid Finding:**
   - Contour detection (find largest quadrilateral)
   - Hough line transform (find grid lines)
   - Perspective warp (flatten the grid to square)

3. **Cell Extraction:**
   - Divide corrected image into 81 cells
   - Crop with margin to avoid grid lines
   - Output: 81 individual cell images

**Libraries:**
- OpenCV (C++/Python/JS bindings available)
- For web: OpenCV.js or custom WebGL shaders

### Phase 2: Digit Recognition (ML)

CNN to classify each cell as 0-9 (0 = empty).

**Model Architecture:**

```
Input: 28x28 grayscale image (normalized)

Conv2D(32, 3x3, ReLU) -> MaxPool(2x2)
Conv2D(64, 3x3, ReLU) -> MaxPool(2x2)
Flatten -> Dense(128, ReLU) -> Dropout(0.5)
Dense(10, Softmax) -> Output: digit 0-9

Parameters: ~100K (small enough for mobile)
```

**Training Data Options:**

| Dataset | Size | Pros | Cons |
|---------|------|------|------|
| MNIST | 70K | Easy baseline | Handwritten only, not printed |
| Printed Digits | Custom | Matches real puzzles | Need to collect/generate |
| Char74K | 74K | Varied fonts | Noisy, needs filtering |
| Synthetic | Unlimited | Perfect control | May not generalize |

**Recommended approach:**
1. Start with MNIST to validate pipeline
2. Collect 1000+ real sudoku images from newspapers/books
3. Augment heavily (rotation, blur, lighting, perspective)
4. Fine-tune on real data
5. Add "empty cell" class (critical - most cells are empty)

**Training Infrastructure:**
- Framework: PyTorch or TensorFlow (PyTorch for flexibility)
- Training time: ~30 min on GPU for this size model
- Export to: ONNX (universal), TFLite (mobile), TF.js (web)

**Expected Accuracy:**
- MNIST baseline: 99%+
- Real printed digits: 95-98% with good training data
- Per-puzzle accuracy: (0.97)^81 = ~8% if independent - validation critical

**Critical Insight:** Post-processing validation required:
- Check row/column/box constraints after recognition
- Flag low-confidence cells for user correction
- Use solver's constraint propagation to eliminate impossible digits

### Phase 3: Solver Integration

**Options:**

| Approach | Effort | Performance | Portability |
|----------|--------|-------------|-------------|
| Port C to WASM | Low | Excellent | Universal |
| Rewrite in Rust -> WASM | Medium | Excellent | Universal |
| Rewrite in JS/TS | Low | Good | Web-native |
| Keep C, use as backend | Medium | Excellent | Server-only |

**Recommendation:** Compile C solver to WebAssembly using Emscripten.

**Solver Enhancements to Consider:**
1. Constraint propagation (naked singles) - faster for easy puzzles
2. Dancing Links (DLX) - fastest for hard puzzles
3. Timeout handling - abort if puzzle is malformed
4. Multiple solution detection - flag if puzzle is ambiguous

---

## Development Phases

### Phase 0: Foundation
**Goal:** Clean up existing code, add file I/O, prepare for integration

Tasks:
- [ ] Commit pending changes
- [ ] Add puzzle validation (reject invalid inputs)
- [ ] Add file I/O (read/write .sudoku files)
- [ ] Add unit tests (validate solver correctness)
- [ ] Create benchmark suite (time various puzzle difficulties)
- [ ] Compile to WASM as proof of concept

### Phase 1: Digit Recognition MVP
**Goal:** Train a working digit classifier

Tasks:
- [ ] Set up Python ML environment (PyTorch, OpenCV)
- [ ] Create training pipeline with MNIST
- [ ] Build data augmentation pipeline
- [ ] Train baseline CNN model
- [ ] Export to ONNX/TFLite/TF.js
- [ ] Create test harness with real sudoku images
- [ ] Measure accuracy, iterate on model

Deliverable: Model achieving >95% accuracy on printed digits

### Phase 2: Grid Detection Pipeline
**Goal:** Extract cells from photos

Tasks:
- [ ] Implement preprocessing (grayscale, threshold, blur)
- [ ] Implement contour detection for grid finding
- [ ] Implement perspective correction
- [ ] Implement cell extraction
- [ ] Handle edge cases (partial grids, tilted photos, poor lighting)
- [ ] Create test suite with varied real-world images

Deliverable: Pipeline that extracts 81 cell images from a photo

### Phase 3: End-to-End Integration (CLI)
**Goal:** Photo in -> solution out (command line)

Tasks:
- [ ] Connect grid detection -> digit recognition -> solver
- [ ] Implement confidence scoring and validation
- [ ] Add constraint-based error correction
- [ ] Benchmark full pipeline latency
- [ ] Create comprehensive test suite

Deliverable: CLI tool that solves sudoku from image files

### Phase 4: Web Application MVP
**Goal:** Browser-based scanner

Tasks:
- [ ] Set up web project (Vite + TypeScript)
- [ ] Integrate TensorFlow.js or ONNX Runtime Web
- [ ] Implement camera capture (MediaDevices API)
- [ ] Port grid detection to JavaScript/WebGL
- [ ] Integrate WASM solver
- [ ] Build minimal UI (capture -> solve -> display)
- [ ] Add manual correction UI for low-confidence cells

Deliverable: Working web app, deployable to static hosting

### Phase 5: Polish and Mobile
**Goal:** Production-ready with mobile support

Tasks:
- [ ] Optimize for mobile browsers
- [ ] Add PWA support (offline, install prompt)
- [ ] Implement AR overlay (show solution on camera feed)
- [ ] Add history/favorites
- [ ] Wrap with Capacitor for app stores (optional)
- [ ] Performance optimization (WebGL, SIMD)

---

## Technical Stack

```
+----------------------------------------------------------+
|                    Frontend (Web)                         |
|  TypeScript + Vite + React/Solid                         |
|  TensorFlow.js (digit recognition)                       |
|  OpenCV.js or custom (grid detection)                    |
|  WASM (C solver compiled with Emscripten)                |
+----------------------------------------------------------+

+----------------------------------------------------------+
|                ML Training Pipeline                       |
|  Python + PyTorch                                         |
|  OpenCV for data preprocessing                           |
|  Weights & Biases for experiment tracking                |
|  Export: ONNX -> TF.js conversion                        |
+----------------------------------------------------------+

+----------------------------------------------------------+
|              Core Solver (C code)                         |
|  Enhanced with validation + constraints                   |
|  Compiled to WASM via Emscripten                         |
|  Also usable standalone for testing                      |
+----------------------------------------------------------+
```

---

## ML Training Data Strategy

### Synthetic Data Generation
```python
# Generate training data programmatically
for font in ["Arial", "Times", "Helvetica", ...]:
    for digit in range(1, 10):
        for rotation in [-15, -10, -5, 0, 5, 10, 15]:
            for blur in [0, 1, 2]:
                for noise in [0, 0.1, 0.2]:
                    generate_cell_image(digit, font, rotation, blur, noise)
```

### Real Data Collection
1. Buy 10-20 sudoku puzzle books (varied publishers)
2. Photograph pages under different lighting
3. Run through grid detection
4. Manually label extracted cells (crowdsource or do yourself)
5. Target: 5000+ real cell images

### Data Augmentation (Critical)
- Random rotation (-20 to +20 degrees)
- Random perspective warp
- Brightness/contrast variation
- Gaussian blur
- Salt and pepper noise
- Random cropping
- Elastic deformation

---

## Risk Analysis

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Grid detection fails on real photos | High | High | Extensive test suite, manual fallback UI |
| Digit recognition accuracy too low | Medium | High | More training data, user correction UI |
| WASM solver too slow | Low | Medium | Optimize algorithm, use Web Workers |
| Mobile camera APIs unreliable | Medium | Medium | Progressive enhancement, file upload fallback |
| Model too large for web | Low | Medium | Quantization, model pruning |

---

## Open Questions

1. **Platform priority:** Web-first, or native mobile from the start?
2. **Offline requirement:** Must work without internet, or server-side inference acceptable?
3. **Scope of "AI":** Just digit recognition, or ML-based solver too?
4. **AR interest:** Solution overlaid on camera feed in real-time, or static result page?

---

## References

- [OpenCV Documentation](https://docs.opencv.org/)
- [TensorFlow.js](https://www.tensorflow.org/js)
- [Emscripten](https://emscripten.org/)
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
