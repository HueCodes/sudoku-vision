/**
 * Grid detection module.
 * Finds the sudoku grid in an image and returns corner points.
 */

import type { Corners, Point, GridDetectionResult } from './types';

declare const cv: any;

/**
 * Preprocess image for grid detection.
 * Converts to grayscale, blurs, and applies adaptive threshold.
 */
function preprocessForGridDetection(src: any): any {
  const gray = new cv.Mat();
  const blurred = new cv.Mat();
  const binary = new cv.Mat();

  // Convert to grayscale
  if (src.channels() === 4) {
    cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
  } else if (src.channels() === 3) {
    cv.cvtColor(src, gray, cv.COLOR_RGB2GRAY);
  } else {
    src.copyTo(gray);
  }

  // Gaussian blur to reduce noise
  cv.GaussianBlur(gray, blurred, new cv.Size(5, 5), 0);

  // Adaptive threshold (binary inverse for white lines on black)
  cv.adaptiveThreshold(
    blurred,
    binary,
    255,
    cv.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv.THRESH_BINARY_INV,
    11,
    2
  );

  gray.delete();
  blurred.delete();

  return binary;
}

/**
 * Order 4 points as: top-left, top-right, bottom-right, bottom-left.
 * This consistent ordering is required for perspective transform.
 */
function orderPoints(pts: Point[]): Corners {
  // Sum: top-left has smallest, bottom-right has largest
  const sums = pts.map((p) => p.x + p.y);
  const tlIdx = sums.indexOf(Math.min(...sums));
  const brIdx = sums.indexOf(Math.max(...sums));

  // Diff: top-right has smallest (y-x), bottom-left has largest
  const diffs = pts.map((p) => p.y - p.x);
  const trIdx = diffs.indexOf(Math.min(...diffs));
  const blIdx = diffs.indexOf(Math.max(...diffs));

  return [pts[tlIdx], pts[trIdx], pts[brIdx], pts[blIdx]];
}

/**
 * Find the sudoku grid contour (largest quadrilateral).
 */
export function detectGrid(canvas: HTMLCanvasElement): GridDetectionResult | null {
  const src = cv.imread(canvas);
  const binary = preprocessForGridDetection(src);

  // Find contours
  const contours = new cv.MatVector();
  const hierarchy = new cv.Mat();
  cv.findContours(binary, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

  const imageArea = binary.rows * binary.cols;
  const minArea = 0.1 * imageArea;

  // Collect contours with their areas
  const contourData: { idx: number; area: number }[] = [];
  for (let i = 0; i < contours.size(); i++) {
    const area = cv.contourArea(contours.get(i));
    if (area >= minArea) {
      contourData.push({ idx: i, area });
    }
  }

  // Sort by area descending
  contourData.sort((a, b) => b.area - a.area);

  let result: GridDetectionResult | null = null;

  for (const { idx, area } of contourData) {
    const contour = contours.get(idx);
    const perimeter = cv.arcLength(contour, true);
    const approx = new cv.Mat();

    cv.approxPolyDP(contour, approx, 0.02 * perimeter, true);

    // Looking for quadrilateral
    if (approx.rows === 4) {
      const points: Point[] = [];
      for (let j = 0; j < 4; j++) {
        points.push({
          x: approx.data32S[j * 2],
          y: approx.data32S[j * 2 + 1],
        });
      }

      const corners = orderPoints(points);
      const confidence = area / imageArea;

      result = { corners, confidence };
      approx.delete();
      break;
    }

    approx.delete();
  }

  // Cleanup
  src.delete();
  binary.delete();
  contours.delete();
  hierarchy.delete();

  return result;
}

/**
 * Draw grid overlay on canvas for visual feedback.
 */
export function drawGridOverlay(
  ctx: CanvasRenderingContext2D,
  corners: Corners,
  scaleX: number,
  scaleY: number
): void {
  ctx.strokeStyle = '#00d4ff';
  ctx.lineWidth = 3;
  ctx.beginPath();

  const scaledCorners = corners.map((p) => ({
    x: p.x * scaleX,
    y: p.y * scaleY,
  }));

  ctx.moveTo(scaledCorners[0].x, scaledCorners[0].y);
  ctx.lineTo(scaledCorners[1].x, scaledCorners[1].y);
  ctx.lineTo(scaledCorners[2].x, scaledCorners[2].y);
  ctx.lineTo(scaledCorners[3].x, scaledCorners[3].y);
  ctx.closePath();
  ctx.stroke();

  // Draw corner circles
  ctx.fillStyle = '#00d4ff';
  for (const corner of scaledCorners) {
    ctx.beginPath();
    ctx.arc(corner.x, corner.y, 8, 0, 2 * Math.PI);
    ctx.fill();
  }
}
