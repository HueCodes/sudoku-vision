/**
 * Cell preprocessing for ML inference.
 * Must match the Python training preprocessing pipeline.
 */

declare const cv: any;

/**
 * Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
 * Matches: cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
 */
function applyCLAHE(grayMat: any): any {
  const clahe = new cv.CLAHE(2.0, new cv.Size(4, 4));
  const result = new cv.Mat();
  clahe.apply(grayMat, result);
  clahe.delete();
  return result;
}

/**
 * Apply adaptive thresholding.
 * Matches: cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
 */
function applyAdaptiveThreshold(grayMat: any): any {
  const result = new cv.Mat();
  cv.adaptiveThreshold(
    grayMat,
    result,
    255,
    cv.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv.THRESH_BINARY,
    11,
    2
  );
  return result;
}

/**
 * Preprocess a cell image for the CNN model.
 * Applies: CLAHE -> Adaptive Threshold -> Invert -> Normalize to [-1, 1]
 *
 * Must match Python training pipeline:
 *   tensor = torch.from_numpy(processed).float() / 255.0
 *   tensor = (tensor - 0.5) / 0.5  # Normalize to [-1, 1]
 *
 * @param imageData - The cell ImageData (28x28 RGBA)
 * @returns Float32Array of shape [1, 28, 28] normalized to [-1, 1]
 */
export function preprocessCell(imageData: ImageData): Float32Array {
  const { width, height } = imageData;

  // Create grayscale mat from ImageData
  const rgba = cv.matFromImageData(imageData);
  const gray = new cv.Mat();
  cv.cvtColor(rgba, gray, cv.COLOR_RGBA2GRAY);

  // Apply CLAHE
  const claheResult = applyCLAHE(gray);

  // Apply adaptive threshold
  const thresholded = applyAdaptiveThreshold(claheResult);

  // Invert (255 - img) - white digit on black background
  const inverted = new cv.Mat();
  cv.bitwise_not(thresholded, inverted);

  // Convert to Float32Array normalized to [-1, 1]
  // Formula: (value / 255.0 - 0.5) / 0.5 = value / 127.5 - 1.0
  const output = new Float32Array(width * height);
  for (let i = 0; i < width * height; i++) {
    output[i] = inverted.data[i] / 127.5 - 1.0;
  }

  // Cleanup
  rgba.delete();
  gray.delete();
  claheResult.delete();
  thresholded.delete();
  inverted.delete();

  return output;
}

/**
 * Simplified preprocessing without OpenCV.
 * Used as fallback if OpenCV CLAHE is not available.
 */
export function preprocessCellSimple(imageData: ImageData): Float32Array {
  const { width, height, data } = imageData;
  const output = new Float32Array(width * height);

  // Convert to grayscale and normalize
  for (let i = 0; i < width * height; i++) {
    const r = data[i * 4];
    const g = data[i * 4 + 1];
    const b = data[i * 4 + 2];

    // Grayscale
    let gray = 0.299 * r + 0.587 * g + 0.114 * b;

    // Simple inversion (assuming light background)
    gray = 255 - gray;

    // Normalize to [-1, 1]
    output[i] = gray / 127.5 - 1.0;
  }

  return output;
}
