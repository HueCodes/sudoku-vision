import CoreGraphics
import CoreImage

/// Extracted cell data ready for digit classification
struct ExtractedCell: Sendable {
    let row: Int
    let col: Int
    let pixels: [Float]  // 28x28 normalized grayscale pixels [0, 1]
}

/// Result of cell extraction
struct ExtractedCells: Sendable {
    let cells: [ExtractedCell]

    /// Get cell at specific position
    func cell(row: Int, col: Int) -> ExtractedCell? {
        cells.first { $0.row == row && $0.col == col }
    }
}

/// Extracts individual cells from a perspective-corrected grid image
final class CellExtractor {
    private let context: CIContext
    private let cellSize = 28  // Output size for ML model
    private let marginRatio: CGFloat = 0.1  // 10% margin to avoid grid lines

    init() {
        context = CIContext(options: [.useSoftwareRenderer: false])
    }

    /// Extract all 81 cells from a warped grid image
    /// - Parameter warpedGrid: Perspective-corrected square grid image
    /// - Returns: 81 extracted cells with normalized pixel data
    func extractCells(from warpedGrid: CIImage) -> ExtractedCells {
        let gridSize = warpedGrid.extent.size
        let cellWidth = gridSize.width / 9
        let cellHeight = gridSize.height / 9

        var cells: [ExtractedCell] = []

        for row in 0..<9 {
            for col in 0..<9 {
                // Calculate cell bounds with margin
                let marginX = cellWidth * marginRatio
                let marginY = cellHeight * marginRatio

                let x = CGFloat(col) * cellWidth + marginX
                let y = CGFloat(8 - row) * cellHeight + marginY  // CIImage is bottom-left origin
                let w = cellWidth - (2 * marginX)
                let h = cellHeight - (2 * marginY)

                let cellRect = CGRect(x: x, y: y, width: w, height: h)

                // Crop cell from grid
                let cropped = warpedGrid.cropped(to: cellRect)

                // Extract and normalize pixels
                if let pixels = extractPixels(from: cropped) {
                    cells.append(ExtractedCell(row: row, col: col, pixels: pixels))
                }
            }
        }

        return ExtractedCells(cells: cells)
    }

    /// Extract normalized pixel data from a cell image
    private func extractPixels(from cellImage: CIImage) -> [Float]? {
        // Scale to 28x28
        let scaleX = CGFloat(cellSize) / cellImage.extent.width
        let scaleY = CGFloat(cellSize) / cellImage.extent.height
        var scaled = cellImage.transformed(by: CGAffineTransform(scaleX: scaleX, y: scaleY))

        // Translate to origin
        scaled = scaled.transformed(
            by: CGAffineTransform(translationX: -scaled.extent.origin.x, y: -scaled.extent.origin.y)
        )

        // Convert to grayscale
        if let grayscale = CIFilter(name: "CIPhotoEffectMono") {
            grayscale.setValue(scaled, forKey: kCIInputImageKey)
            if let output = grayscale.outputImage {
                scaled = output
            }
        }

        // Render to CGImage
        guard let cgImage = context.createCGImage(scaled, from: CGRect(x: 0, y: 0, width: cellSize, height: cellSize)) else {
            return nil
        }

        // Extract pixel data
        return extractGrayscalePixels(from: cgImage)
    }

    /// Extract grayscale pixel values normalized to [0, 1]
    private func extractGrayscalePixels(from cgImage: CGImage) -> [Float]? {
        let width = cgImage.width
        let height = cgImage.height

        guard width == cellSize && height == cellSize else {
            return nil
        }

        // Create grayscale context
        let colorSpace = CGColorSpaceCreateDeviceGray()
        var pixelData = [UInt8](repeating: 0, count: width * height)

        guard let cgContext = CGContext(
            data: &pixelData,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.none.rawValue
        ) else {
            return nil
        }

        cgContext.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))

        // Normalize to [0, 1] and invert (white background, black digits -> black background, white digits)
        // This matches typical MNIST format
        return pixelData.map { 1.0 - Float($0) / 255.0 }
    }

    /// Check if a cell is likely empty (mostly white/background)
    func isLikelyEmpty(_ cell: ExtractedCell, threshold: Float = 0.15) -> Bool {
        // Calculate mean pixel intensity
        let mean = cell.pixels.reduce(0, +) / Float(cell.pixels.count)
        // Low mean = mostly dark (inverted) = mostly background = empty
        return mean < threshold
    }

    /// Calculate variance of pixel intensities (useful for empty detection)
    func pixelVariance(_ cell: ExtractedCell) -> Float {
        let mean = cell.pixels.reduce(0, +) / Float(cell.pixels.count)
        let variance = cell.pixels.map { pow($0 - mean, 2) }.reduce(0, +) / Float(cell.pixels.count)
        return variance
    }
}
