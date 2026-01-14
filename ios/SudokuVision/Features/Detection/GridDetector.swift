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
actor GridDetector {

    /// Detect the largest square-like rectangle in the image
    func detectGrid(in pixelBuffer: CVPixelBuffer) async -> DetectedGrid? {
        let request = VNDetectRectanglesRequest()
        request.minimumAspectRatio = 0.8
        request.maximumAspectRatio = 1.2
        request.minimumSize = 0.3 // At least 30% of image
        request.maximumObservations = 5
        request.minimumConfidence = 0.5

        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])

        do {
            try handler.perform([request])
        } catch {
            return nil
        }

        guard let results = request.results, !results.isEmpty else {
            return nil
        }

        // Find the largest rectangle that's closest to square
        let best = results
            .filter { isNearlySquare($0) }
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
