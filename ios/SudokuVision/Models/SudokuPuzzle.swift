import Foundation

/// Domain model representing a scanned sudoku puzzle
struct SudokuPuzzle: Sendable {
    let original: SudokuGrid
    let solution: SudokuGrid?
    let confidences: [[Float]]
    let capturedAt: Date

    init(original: SudokuGrid, solution: SudokuGrid? = nil, confidences: [[Float]]? = nil) {
        self.original = original
        self.solution = solution
        self.confidences = confidences ?? Array(repeating: Array(repeating: 1.0, count: 9), count: 9)
        self.capturedAt = Date()
    }

    /// Cells where recognition confidence is below threshold
    func lowConfidenceCells(threshold: Float = 0.7) -> [(row: Int, col: Int)] {
        var result: [(row: Int, col: Int)] = []
        for row in 0..<9 {
            for col in 0..<9 {
                if confidences[row][col] < threshold && original[row, col] != 0 {
                    result.append((row, col))
                }
            }
        }
        return result
    }

    /// Average confidence across all recognized digits
    var averageConfidence: Float {
        var total: Float = 0
        var count = 0
        for row in 0..<9 {
            for col in 0..<9 {
                if original[row, col] != 0 {
                    total += confidences[row][col]
                    count += 1
                }
            }
        }
        return count > 0 ? total / Float(count) : 1.0
    }
}
