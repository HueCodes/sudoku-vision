import Accelerate
import CoreGraphics
import Foundation

/// Preprocesses cell images to match the Python training pipeline.
///
/// This preprocessor implements CLAHE-equivalent contrast enhancement and
/// adaptive thresholding to match the preprocessing used during model training.
/// Without this preprocessing, accuracy drops significantly on real printed digits.
///
/// ## Python Reference (from ml/datasets.py)
/// ```python
/// clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
/// img = clahe.apply(img)
/// img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
/// img = 255 - img  # Invert
/// img = img.astype(np.float32) / 127.5 - 1.0  # Normalize to [-1, 1]
/// ```
enum CellPreprocessor {

    // MARK: - Configuration

    /// CLAHE clip limit (matches Python clipLimit=2.0)
    private static let claheClipLimit: Float = 2.0

    /// CLAHE tile grid size (matches Python tileGridSize=(4,4) for 28x28 images)
    /// For 28x28, we use 7x7 tiles (4 tiles per dimension)
    private static let claheTileSize: Int = 7

    /// Adaptive threshold block size (matches Python blockSize=11)
    private static let adaptiveBlockSize: Int = 11

    /// Adaptive threshold constant (matches Python C=2)
    private static let adaptiveC: Float = 2.0

    // MARK: - Main Preprocessing

    /// Preprocess cell pixels to match training pipeline.
    ///
    /// Applies: CLAHE → Adaptive Threshold → Invert → Normalize to [-1, 1]
    ///
    /// - Parameter pixels: 28x28 raw grayscale pixels in [0, 1] range
    /// - Returns: Preprocessed pixels in [-1, 1] range
    static func preprocess(_ pixels: [Float]) -> [Float] {
        guard pixels.count == 28 * 28 else {
            return pixels
        }

        // Convert to UInt8 for histogram processing
        var uint8Pixels = pixels.map { UInt8(min(255, max(0, $0 * 255))) }

        // Step 1: Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        uint8Pixels = applyCLAHE(uint8Pixels, width: 28, height: 28)

        // Step 2: Apply adaptive thresholding
        uint8Pixels = applyAdaptiveThreshold(uint8Pixels, width: 28, height: 28)

        // Step 3: Invert (white digit on black background)
        uint8Pixels = uint8Pixels.map { 255 - $0 }

        // Step 4: Normalize to [-1, 1] - matching model training (Normalize((0.5,), (0.5,)))
        return uint8Pixels.map { Float($0) / 127.5 - 1.0 }
    }

    // MARK: - CLAHE Implementation

    /// Apply Contrast Limited Adaptive Histogram Equalization.
    ///
    /// This is a simplified CLAHE implementation that divides the image into tiles
    /// and applies histogram equalization to each tile with contrast limiting.
    private static func applyCLAHE(_ pixels: [UInt8], width: Int, height: Int) -> [UInt8] {
        var result = pixels

        let tilesX = width / claheTileSize
        let tilesY = height / claheTileSize

        // Process each tile
        for ty in 0..<tilesY {
            for tx in 0..<tilesX {
                let startX = tx * claheTileSize
                let startY = ty * claheTileSize

                // Extract tile pixels
                var tilePixels = [UInt8]()
                for y in startY..<min(startY + claheTileSize, height) {
                    for x in startX..<min(startX + claheTileSize, width) {
                        tilePixels.append(pixels[y * width + x])
                    }
                }

                // Compute histogram
                var histogram = [Int](repeating: 0, count: 256)
                for pixel in tilePixels {
                    histogram[Int(pixel)] += 1
                }

                // Apply contrast limiting
                let clipLimit = Int(claheClipLimit * Float(tilePixels.count) / 256.0)
                var excess = 0
                for i in 0..<256 {
                    if histogram[i] > clipLimit {
                        excess += histogram[i] - clipLimit
                        histogram[i] = clipLimit
                    }
                }

                // Redistribute excess
                let redistribution = excess / 256
                for i in 0..<256 {
                    histogram[i] += redistribution
                }

                // Compute CDF (cumulative distribution function)
                var cdf = [Int](repeating: 0, count: 256)
                cdf[0] = histogram[0]
                for i in 1..<256 {
                    cdf[i] = cdf[i - 1] + histogram[i]
                }

                // Find min CDF value
                var cdfMin = 0
                for i in 0..<256 {
                    if cdf[i] > 0 {
                        cdfMin = cdf[i]
                        break
                    }
                }

                // Create lookup table for histogram equalization
                var lut = [UInt8](repeating: 0, count: 256)
                let denominator = max(1, tilePixels.count - cdfMin)
                for i in 0..<256 {
                    let value = Float(cdf[i] - cdfMin) / Float(denominator) * 255.0
                    lut[i] = UInt8(min(255, max(0, value)))
                }

                // Apply LUT to tile pixels in result
                for y in startY..<min(startY + claheTileSize, height) {
                    for x in startX..<min(startX + claheTileSize, width) {
                        let idx = y * width + x
                        result[idx] = lut[Int(pixels[idx])]
                    }
                }
            }
        }

        return result
    }

    // MARK: - Adaptive Thresholding

    /// Apply adaptive thresholding using local mean.
    ///
    /// For each pixel, threshold = local_mean - C
    /// Pixel becomes 255 if value > threshold, else 0.
    private static func applyAdaptiveThreshold(_ pixels: [UInt8], width: Int, height: Int) -> [UInt8] {
        var result = [UInt8](repeating: 0, count: pixels.count)
        let halfBlock = adaptiveBlockSize / 2

        for y in 0..<height {
            for x in 0..<width {
                // Calculate local mean in block
                var sum: Float = 0
                var count: Float = 0

                for dy in -halfBlock...halfBlock {
                    for dx in -halfBlock...halfBlock {
                        let ny = y + dy
                        let nx = x + dx

                        if ny >= 0 && ny < height && nx >= 0 && nx < width {
                            sum += Float(pixels[ny * width + nx])
                            count += 1
                        }
                    }
                }

                let localMean = sum / count
                let threshold = localMean - adaptiveC

                let pixelValue = Float(pixels[y * width + x])
                result[y * width + x] = pixelValue > threshold ? 255 : 0
            }
        }

        return result
    }

    // MARK: - Fast Preprocessing (Using Accelerate)

    /// Alternative faster preprocessing using vDSP.
    ///
    /// This is a simplified version that applies contrast stretch and threshold
    /// without full CLAHE, but is much faster. Use for real-time processing.
    static func preprocessFast(_ pixels: [Float]) -> [Float] {
        guard pixels.count == 28 * 28 else {
            return pixels
        }

        var result = pixels

        // Find min/max for contrast stretch
        var minVal: Float = 0
        var maxVal: Float = 0
        vDSP_minv(pixels, 1, &minVal, vDSP_Length(pixels.count))
        vDSP_maxv(pixels, 1, &maxVal, vDSP_Length(pixels.count))

        // Contrast stretch to [0, 1]
        let range = max(maxVal - minVal, 0.001)
        var negMin = -minVal
        vDSP_vsadd(pixels, 1, &negMin, &result, 1, vDSP_Length(pixels.count))
        var invRange = 1.0 / range
        vDSP_vsmul(result, 1, &invRange, &result, 1, vDSP_Length(pixels.count))

        // Apply threshold at 0.5
        result = result.map { $0 > 0.5 ? 0.0 : 1.0 }  // Inverted binary

        // Normalize to [-1, 1]
        var scale: Float = 2.0
        var offset: Float = -1.0
        vDSP_vsmsa(result, 1, &scale, &offset, &result, 1, vDSP_Length(result.count))

        return result
    }
}
