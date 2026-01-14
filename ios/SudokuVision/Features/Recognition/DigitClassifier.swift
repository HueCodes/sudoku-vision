import Foundation

/// Result of classifying a single cell
struct DigitPrediction: Sendable, Equatable {
    let digit: Int        // 0-9 (0 = empty cell)
    let confidence: Float // 0.0-1.0

    static let empty = DigitPrediction(digit: 0, confidence: 1.0)
}

/// Result of classifying all 81 cells
struct ClassificationResult: Sendable {
    let predictions: [[DigitPrediction]]  // 9x9 grid of predictions
    let grid: SudokuGrid
    let confidences: [[Float]]

    init(predictions: [[DigitPrediction]]) {
        self.predictions = predictions

        // Build grid from predictions
        var grid = SudokuGrid()
        var confidences = [[Float]](repeating: [Float](repeating: 0, count: 9), count: 9)

        for row in 0..<9 {
            for col in 0..<9 {
                let pred = predictions[row][col]
                grid[row, col] = pred.digit
                confidences[row][col] = pred.confidence
            }
        }

        self.grid = grid
        self.confidences = confidences
    }

    /// Get cells with confidence below threshold
    func lowConfidenceCells(threshold: Float = 0.7) -> [(row: Int, col: Int, prediction: DigitPrediction)] {
        var result: [(row: Int, col: Int, prediction: DigitPrediction)] = []
        for row in 0..<9 {
            for col in 0..<9 {
                let pred = predictions[row][col]
                if pred.confidence < threshold && pred.digit != 0 {
                    result.append((row, col, pred))
                }
            }
        }
        return result
    }

    /// Average confidence across non-empty cells
    var averageConfidence: Float {
        var total: Float = 0
        var count = 0
        for row in 0..<9 {
            for col in 0..<9 {
                if predictions[row][col].digit != 0 {
                    total += predictions[row][col].confidence
                    count += 1
                }
            }
        }
        return count > 0 ? total / Float(count) : 1.0
    }
}

/// Protocol for digit classification implementations
protocol DigitClassifierProtocol: Sendable {
    /// Classify all 81 cells
    func classify(cells: ExtractedCells) async -> ClassificationResult

    /// Classify a single cell
    func classifyCell(_ cell: ExtractedCell) async -> DigitPrediction
}
