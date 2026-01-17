/**
 * Cell extraction module.
 * Extracts 81 individual cells from a warped grid image.
 */

import type { ProcessedCell } from './types';

declare const cv: any;

/**
 * Check if a cell appears empty (no digit).
 */
function isCellEmpty(cellMat: any, threshold: number = 0.02): boolean {
  const binary = new cv.Mat();

  // Otsu threshold
  cv.threshold(cellMat, binary, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU);

  // Count non-zero pixels
  const nonZero = cv.countNonZero(binary);
  const total = binary.rows * binary.cols;

  binary.delete();

  return (nonZero / total) < threshold;
}

/**
 * Extract 81 cell images from warped grid.
 */
export function extractCells(
  warpedCanvas: HTMLCanvasElement,
  cellSize: number = 28,
  marginRatio: number = 0.15
): ProcessedCell[] {
  const src = cv.imread(warpedCanvas);

  // Convert to grayscale if needed
  let gray: any;
  if (src.channels() === 4) {
    gray = new cv.Mat();
    cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
  } else if (src.channels() === 3) {
    gray = new cv.Mat();
    cv.cvtColor(src, gray, cv.COLOR_RGB2GRAY);
  } else {
    gray = src.clone();
  }

  const h = gray.rows;
  const w = gray.cols;
  const cellH = Math.floor(h / 9);
  const cellW = Math.floor(w / 9);
  const marginH = Math.floor(cellH * marginRatio);
  const marginW = Math.floor(cellW * marginRatio);

  const cells: ProcessedCell[] = [];

  // Temporary canvas for cell output
  const cellCanvas = document.createElement('canvas');
  cellCanvas.width = cellSize;
  cellCanvas.height = cellSize;

  for (let row = 0; row < 9; row++) {
    for (let col = 0; col < 9; col++) {
      // Cell bounds with margin
      const y1 = row * cellH + marginH;
      const y2 = (row + 1) * cellH - marginH;
      const x1 = col * cellW + marginW;
      const x2 = (col + 1) * cellW - marginW;

      // Extract cell region
      const cellRect = new cv.Rect(x1, y1, x2 - x1, y2 - y1);
      const cellMat = gray.roi(cellRect);

      // Resize to target size
      const resized = new cv.Mat();
      cv.resize(cellMat, resized, new cv.Size(cellSize, cellSize));

      // Check if empty
      const isEmpty = isCellEmpty(resized);

      // Convert to ImageData
      cv.imshow(cellCanvas, resized);
      const ctx = cellCanvas.getContext('2d')!;
      const imageData = ctx.getImageData(0, 0, cellSize, cellSize);

      cells.push({
        imageData,
        row,
        col,
        isEmpty,
      });

      cellMat.delete();
      resized.delete();
    }
  }

  src.delete();
  gray.delete();

  return cells;
}

/**
 * Convert cell ImageData to grayscale Float32Array for ML input.
 * Applies preprocessing to match training pipeline.
 */
export function preprocessCellForModel(imageData: ImageData): Float32Array {
  const { width, height, data } = imageData;
  const output = new Float32Array(width * height);

  // Convert to grayscale (already grayscale from extraction, but handle RGBA)
  for (let i = 0; i < width * height; i++) {
    const r = data[i * 4];
    const g = data[i * 4 + 1];
    const b = data[i * 4 + 2];
    // Grayscale conversion
    output[i] = 0.299 * r + 0.587 * g + 0.114 * b;
  }

  return output;
}
