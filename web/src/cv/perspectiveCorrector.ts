/**
 * Perspective correction module.
 * Warps detected grid region to a square image.
 */

import type { Corners } from './types';

declare const cv: any;

/**
 * Warp the grid region to a square image.
 */
export function warpPerspective(
  canvas: HTMLCanvasElement,
  corners: Corners,
  outputSize: number = 450
): HTMLCanvasElement {
  const src = cv.imread(canvas);
  const dst = new cv.Mat();

  // Source points (detected corners)
  const srcPoints = cv.matFromArray(4, 1, cv.CV_32FC2, [
    corners[0].x, corners[0].y,  // TL
    corners[1].x, corners[1].y,  // TR
    corners[2].x, corners[2].y,  // BR
    corners[3].x, corners[3].y,  // BL
  ]);

  // Destination points (square)
  const dstPoints = cv.matFromArray(4, 1, cv.CV_32FC2, [
    0, 0,                          // TL
    outputSize - 1, 0,             // TR
    outputSize - 1, outputSize - 1, // BR
    0, outputSize - 1,             // BL
  ]);

  // Get perspective transform matrix
  const matrix = cv.getPerspectiveTransform(srcPoints, dstPoints);

  // Apply warp
  cv.warpPerspective(
    src,
    dst,
    matrix,
    new cv.Size(outputSize, outputSize)
  );

  // Create output canvas
  const outputCanvas = document.createElement('canvas');
  outputCanvas.width = outputSize;
  outputCanvas.height = outputSize;
  cv.imshow(outputCanvas, dst);

  // Cleanup
  src.delete();
  dst.delete();
  srcPoints.delete();
  dstPoints.delete();
  matrix.delete();

  return outputCanvas;
}
