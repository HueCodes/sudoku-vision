import CoreImage
import Vision

/// Detected grid corners in normalized coordinates (0-1)
struct DetectedGrid: Equatable, Sendable {
    let topLeft: CGPoint
    let topRight: CGPoint
    let bottomRight: CGPoint
    let bottomLeft: CGPoint
    let confidence: Float

    /// Convert normalized coordinates to view coordinates
    func points(in size: CGSize) -> [CGPoint] {
        [
            CGPoint(x: topLeft.x * size.width, y: topLeft.y * size.height),
            CGPoint(x: topRight.x * size.width, y: topRight.y * size.height),
            CGPoint(x: bottomRight.x * size.width, y: bottomRight.y * size.height),
            CGPoint(x: bottomLeft.x * size.width, y: bottomLeft.y * size.height)
        ]
    }
}

/// Detects sudoku grids in camera frames using Vision.framework
final class GridDetector: Sendable {

    /// Detect the largest square-like rectangle in the image
    func detectGrid(in pixelBuffer: CVPixelBuffer) -> DetectedGrid? {
        let request = VNDetectRectanglesRequest()
        request.minimumAspectRatio = 0.7  // More lenient
        request.maximumAspectRatio = 1.4  // More lenient
        request.minimumSize = 0.2 // At least 20% of image (was 30%)
        request.maximumObservations = 10
        request.minimumConfidence = 0.3  // Lower threshold (was 0.5)

        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])

        do {
            try handler.perform([request])
        } catch {
            print("[GridDetector] Error: \(error)")
            return nil
        }

        guard let results = request.results, !results.isEmpty else {
            // Uncomment for verbose debugging:
            // print("[GridDetector] No rectangles found")
            return nil
        }

        // Find the largest rectangle that's closest to square
        let squareOnes = results.filter { isNearlySquare($0) }

        let best = squareOnes
            .max { area(of: $0) < area(of: $1) }

        guard let observation = best else {
            return nil
        }

        // Vision uses bottom-left origin, convert to top-left
        return DetectedGrid(
            topLeft: CGPoint(x: observation.topLeft.x, y: 1 - observation.topLeft.y),
            topRight: CGPoint(x: observation.topRight.x, y: 1 - observation.topRight.y),
            bottomRight: CGPoint(x: observation.bottomRight.x, y: 1 - observation.bottomRight.y),
            bottomLeft: CGPoint(x: observation.bottomLeft.x, y: 1 - observation.bottomLeft.y),
            confidence: observation.confidence
        )
    }

    private func isNearlySquare(_ observation: VNRectangleObservation) -> Bool {
        let width = distance(from: observation.topLeft, to: observation.topRight)
        let height = distance(from: observation.topLeft, to: observation.bottomLeft)
        let ratio = min(width, height) / max(width, height)
        return ratio > 0.75
    }

    private func area(of observation: VNRectangleObservation) -> CGFloat {
        let width = distance(from: observation.topLeft, to: observation.topRight)
        let height = distance(from: observation.topLeft, to: observation.bottomLeft)
        return width * height
    }

    private func distance(from p1: CGPoint, to p2: CGPoint) -> CGFloat {
        sqrt(pow(p2.x - p1.x, 2) + pow(p2.y - p1.y, 2))
    }
}
