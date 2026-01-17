/**
 * Main entry point for Sudoku Vision web app.
 * Orchestrates camera, CV pipeline, ML inference, and solver.
 */

import { Camera } from './camera';
import { detectGrid, drawGridOverlay, warpPerspective, extractCells, type Corners } from './cv';
import { DigitClassifier } from './ml';
import { solve, initSolver } from './solver';

// Global state
let camera: Camera | null = null;
let classifier: DigitClassifier | null = null;
let cvReady = false;
let detectedCorners: Corners | null = null;
let animationFrameId: number | null = null;

// DOM elements
const loadingView = document.getElementById('loading-view')!;
const cameraView = document.getElementById('camera-view')!;
const resultsView = document.getElementById('results-view')!;
const errorView = document.getElementById('error-view')!;
const initStatus = document.getElementById('init-status')!;
const video = document.getElementById('video') as HTMLVideoElement;
const canvasOverlay = document.getElementById('canvas-overlay') as HTMLCanvasElement;
const captureBtn = document.getElementById('capture-btn') as HTMLButtonElement;
const scanAgainBtn = document.getElementById('scan-again-btn') as HTMLButtonElement;
const retryBtn = document.getElementById('retry-btn') as HTMLButtonElement;
const processingOverlay = document.getElementById('processing-overlay')!;
const processingStatus = document.getElementById('processing-status')!;
const solutionGrid = document.getElementById('solution-grid')!;
const resultsInfoText = document.getElementById('results-info-text')!;
const errorMessage = document.getElementById('error-message')!;
const debugOutput = document.getElementById('debug-output')!;
const fileInput = document.getElementById('file-input') as HTMLInputElement;

/**
 * Show a specific view.
 */
function showView(view: 'loading' | 'camera' | 'results' | 'error'): void {
  loadingView.classList.remove('active');
  cameraView.classList.remove('active');
  resultsView.classList.remove('active');
  errorView.classList.remove('active');

  switch (view) {
    case 'loading':
      loadingView.classList.add('active');
      break;
    case 'camera':
      cameraView.classList.add('active');
      break;
    case 'results':
      resultsView.classList.add('active');
      break;
    case 'error':
      errorView.classList.add('active');
      break;
  }
}

/**
 * Update initialization status message.
 */
function updateStatus(message: string): void {
  initStatus.textContent = message;
  console.log('[Init]', message);
}

/**
 * Show/hide processing overlay.
 */
function setProcessing(show: boolean, message?: string): void {
  if (show) {
    processingOverlay.classList.add('active');
    if (message) {
      processingStatus.textContent = message;
    }
  } else {
    processingOverlay.classList.remove('active');
  }
}

/**
 * Show error view with message.
 */
function showError(message: string): void {
  errorMessage.textContent = message;
  showView('error');
}

/**
 * Debug log helper.
 */
function debug(message: string): void {
  console.log('[Debug]', message);
  debugOutput.textContent += message + '\n';
}

/**
 * Load OpenCV.js
 */
async function loadOpenCV(): Promise<void> {
  return new Promise((resolve, reject) => {
    if (typeof cv !== 'undefined' && cv.Mat) {
      cvReady = true;
      resolve();
      return;
    }

    const script = document.createElement('script');
    script.src = '/opencv.js';
    script.async = true;

    script.onload = () => {
      // OpenCV.js takes time to initialize
      const checkReady = () => {
        if (typeof cv !== 'undefined' && cv.Mat) {
          cvReady = true;
          resolve();
        } else {
          setTimeout(checkReady, 100);
        }
      };
      checkReady();
    };

    script.onerror = () => {
      reject(new Error('Failed to load OpenCV.js'));
    };

    document.head.appendChild(script);
  });
}

/**
 * Initialize all components.
 */
async function init(): Promise<void> {
  try {
    // Load OpenCV
    updateStatus('Loading OpenCV.js...');
    await loadOpenCV();
    debug('OpenCV loaded');

    // Load ML model
    updateStatus('Loading digit classifier...');
    classifier = new DigitClassifier();
    await classifier.load('/models/digit_cnn_v2.onnx');
    debug('Classifier loaded');

    // Load solver
    updateStatus('Loading solver...');
    await initSolver();
    debug('Solver loaded');

    // Initialize camera
    updateStatus('Initializing camera...');
    camera = new Camera(video);
    await camera.start();
    debug('Camera started');

    // Set up overlay canvas
    const dims = camera.getDimensions();
    canvasOverlay.width = dims.width;
    canvasOverlay.height = dims.height;

    // Start grid detection loop
    startGridDetection();

    // Show camera view
    showView('camera');
    debug('Ready!');
  } catch (error) {
    console.error('Initialization error:', error);
    showError(error instanceof Error ? error.message : 'Failed to initialize');
  }
}

/**
 * Start continuous grid detection for visual feedback.
 */
function startGridDetection(): void {
  const overlayCtx = canvasOverlay.getContext('2d')!;

  const detectLoop = () => {
    if (!camera?.isActive() || !cvReady) {
      animationFrameId = requestAnimationFrame(detectLoop);
      return;
    }

    try {
      // Capture frame
      const frameCanvas = camera.captureCanvas();

      // Detect grid
      const result = detectGrid(frameCanvas);

      // Clear overlay
      overlayCtx.clearRect(0, 0, canvasOverlay.width, canvasOverlay.height);

      if (result) {
        detectedCorners = result.corners;

        // Calculate scale factors for overlay
        const scaleX = canvasOverlay.width / frameCanvas.width;
        const scaleY = canvasOverlay.height / frameCanvas.height;

        // Draw overlay
        drawGridOverlay(overlayCtx, detectedCorners, scaleX, scaleY);
      } else {
        detectedCorners = null;
      }
    } catch (error) {
      console.warn('Grid detection error:', error);
    }

    animationFrameId = requestAnimationFrame(detectLoop);
  };

  detectLoop();
}

/**
 * Stop grid detection loop.
 */
function stopGridDetection(): void {
  if (animationFrameId !== null) {
    cancelAnimationFrame(animationFrameId);
    animationFrameId = null;
  }
}

/**
 * Capture and process the current frame.
 */
async function captureAndProcess(): Promise<void> {
  if (!camera || !classifier || !cvReady) {
    showError('System not ready');
    return;
  }

  if (!detectedCorners) {
    showError('No sudoku grid detected. Please position the puzzle in the frame.');
    return;
  }

  try {
    setProcessing(true, 'Capturing...');
    stopGridDetection();

    // Capture frame
    const frameCanvas = camera.captureCanvas();

    // Warp perspective
    setProcessing(true, 'Correcting perspective...');
    const warpedCanvas = warpPerspective(frameCanvas, detectedCorners, 450);

    // Extract cells
    setProcessing(true, 'Extracting cells...');
    const cells = extractCells(warpedCanvas, 28, 0.15);
    debug(`Extracted ${cells.length} cells, ${cells.filter((c) => c.isEmpty).length} empty`);

    // Classify digits
    setProcessing(true, 'Recognizing digits...');
    const predictions = await classifier.classifyCells(cells);

    // Build grid
    const grid: number[][] = [];
    const originalCells: boolean[][] = [];
    for (let row = 0; row < 9; row++) {
      grid[row] = [];
      originalCells[row] = [];
      for (let col = 0; col < 9; col++) {
        const idx = row * 9 + col;
        const pred = predictions[idx];
        grid[row][col] = pred.digit;
        originalCells[row][col] = pred.digit !== 0;
      }
    }

    // Log recognized grid
    debug('Recognized grid:');
    for (let row = 0; row < 9; row++) {
      debug(grid[row].map((d) => d || '.').join(' '));
    }

    // Solve
    setProcessing(true, 'Solving puzzle...');
    const solution = await solve(grid);

    if (!solution) {
      showError('Could not solve puzzle. Please try again with a clearer image.');
      startGridDetection();
      return;
    }

    // Display solution
    displaySolution(solution, originalCells);
    showView('results');
    setProcessing(false);
  } catch (error) {
    console.error('Processing error:', error);
    showError(error instanceof Error ? error.message : 'Processing failed');
    startGridDetection();
    setProcessing(false);
  }
}

/**
 * Display the solved sudoku grid.
 */
function displaySolution(solution: number[][], originalCells: boolean[][]): void {
  solutionGrid.innerHTML = '';

  for (let row = 0; row < 9; row++) {
    for (let col = 0; col < 9; col++) {
      const cell = document.createElement('div');
      cell.className = 'sudoku-cell';

      const value = solution[row][col];
      cell.textContent = value ? String(value) : '';

      if (originalCells[row][col]) {
        cell.classList.add('original');
      } else if (value) {
        cell.classList.add('solved');
      } else {
        cell.classList.add('empty');
      }

      solutionGrid.appendChild(cell);
    }
  }

  const originalCount = originalCells.flat().filter(Boolean).length;
  resultsInfoText.textContent = `Solved! ${81 - originalCount} cells filled in.`;
}

/**
 * Reset to camera view.
 */
function scanAgain(): void {
  showView('camera');
  startGridDetection();
}

/**
 * Process an uploaded image file.
 */
async function processImageFile(file: File): Promise<void> {
  if (!classifier || !cvReady) {
    showError('System not ready');
    return;
  }

  try {
    setProcessing(true, 'Loading image...');
    stopGridDetection();

    // Load image into canvas
    const img = await loadImageFromFile(file);
    const canvas = document.createElement('canvas');
    canvas.width = img.width;
    canvas.height = img.height;
    const ctx = canvas.getContext('2d')!;
    ctx.drawImage(img, 0, 0);

    // Detect grid
    setProcessing(true, 'Detecting grid...');
    const gridResult = detectGrid(canvas);

    if (!gridResult) {
      showError('No sudoku grid detected in the image.');
      setProcessing(false);
      showView('camera');
      return;
    }

    debug(`Grid detected with confidence: ${(gridResult.confidence * 100).toFixed(1)}%`);

    // Warp perspective
    setProcessing(true, 'Correcting perspective...');
    const warpedCanvas = warpPerspective(canvas, gridResult.corners, 450);

    // Extract cells
    setProcessing(true, 'Extracting cells...');
    const cells = extractCells(warpedCanvas, 28, 0.15);
    debug(`Extracted ${cells.length} cells, ${cells.filter((c) => c.isEmpty).length} empty`);

    // Classify digits
    setProcessing(true, 'Recognizing digits...');
    const predictions = await classifier.classifyCells(cells);

    // Build grid
    const grid: number[][] = [];
    const originalCells: boolean[][] = [];
    for (let row = 0; row < 9; row++) {
      grid[row] = [];
      originalCells[row] = [];
      for (let col = 0; col < 9; col++) {
        const idx = row * 9 + col;
        const pred = predictions[idx];
        grid[row][col] = pred.digit;
        originalCells[row][col] = pred.digit !== 0;
      }
    }

    // Log recognized grid
    debug('Recognized grid:');
    for (let row = 0; row < 9; row++) {
      debug(grid[row].map((d) => d || '.').join(' '));
    }

    // Solve
    setProcessing(true, 'Solving puzzle...');
    const solution = await solve(grid);

    if (!solution) {
      showError('Could not solve puzzle. The recognized digits may be incorrect.');
      setProcessing(false);
      showView('camera');
      return;
    }

    // Display solution
    displaySolution(solution, originalCells);
    showView('results');
    setProcessing(false);
  } catch (error) {
    console.error('Processing error:', error);
    showError(error instanceof Error ? error.message : 'Processing failed');
    setProcessing(false);
    showView('camera');
  }
}

/**
 * Load an image from a File object.
 */
function loadImageFromFile(file: File): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      URL.revokeObjectURL(img.src);
      resolve(img);
    };
    img.onerror = () => {
      URL.revokeObjectURL(img.src);
      reject(new Error('Failed to load image'));
    };
    img.src = URL.createObjectURL(file);
  });
}

// Event listeners
captureBtn.addEventListener('click', captureAndProcess);
scanAgainBtn.addEventListener('click', scanAgain);
retryBtn.addEventListener('click', () => {
  showView('camera');
  startGridDetection();
});
fileInput.addEventListener('change', (e) => {
  const file = (e.target as HTMLInputElement).files?.[0];
  if (file) {
    processImageFile(file);
    fileInput.value = ''; // Reset for next upload
  }
});

// Initialize on load
document.addEventListener('DOMContentLoaded', init);

// Expose for debugging
declare global {
  interface Window {
    cv: any;
  }
}
