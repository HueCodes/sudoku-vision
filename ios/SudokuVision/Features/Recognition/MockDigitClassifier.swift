import CoreVideo
import Foundation

/// Mock digit classifier that uses pixel analysis to detect empty cells
/// and returns mock digits for non-empty cells.
/// Replace with CoreML implementation when model is ready.
actor MockDigitClassifier: DigitClassifierProtocol {

    private let cellExtractor = CellExtractor()

    // Threshold for determining if a cell is empty
    private let emptyThreshold: Float = 0.12
    private let varianceThreshold: Float = 0.02

    /// Classify all 81 cells from extracted cell data
    func classify(cells: ExtractedCells) async -> ClassificationResult {
        var predictions = [[DigitPrediction]](
            repeating: [DigitPrediction](repeating: .empty, count: 9),
            count: 9
        )

        for cell in cells.cells {
            let prediction = await classifyCell(cell)
            predictions[cell.row][cell.col] = prediction
        }

        return ClassificationResult(predictions: predictions)
    }

    /// Classify a single extracted cell
    func classifyCell(_ cell: ExtractedCell) async -> DigitPrediction {
        // Analyze pixel data to determine if cell is empty
        let mean = cell.pixels.reduce(0, +) / Float(cell.pixels.count)
        let variance = cell.pixels.map { pow($0 - mean, 2) }.reduce(0, +) / Float(cell.pixels.count)

        // Empty cell detection:
        // - Low mean intensity (mostly background after inversion)
        // - Low variance (uniform)
        if mean < emptyThreshold || variance < varianceThreshold {
            return DigitPrediction(digit: 0, confidence: 0.95)
        }

        // For non-empty cells, return a mock digit based on cell position
        // This creates a valid-looking puzzle for testing
        // In production, this would run CoreML inference
        let mockDigit = mockDigitForCell(row: cell.row, col: cell.col, mean: mean)
        let confidence = Float.random(in: 0.85...0.98)

        return DigitPrediction(digit: mockDigit, confidence: confidence)
    }

    /// Generate a mock digit that creates a solvable puzzle pattern
    private func mockDigitForCell(row: Int, col: Int, mean: Float) -> Int {
        // Use pixel intensity to generate deterministic but varied digits
        // This ensures the same image produces the same mock result
        let seed = Int(mean * 1000) + row * 10 + col
        let digit = (seed % 9) + 1
        return digit
    }

    // MARK: - Legacy API for backward compatibility

    /// Legacy method: Classify from pixel buffer and detected grid
    /// Now properly extracts cells first
    func classify(pixelBuffer: CVPixelBuffer, grid: DetectedGrid) async -> SudokuGrid {
        // For backward compatibility, return a hardcoded test puzzle
        // The real pipeline should use the new classify(cells:) method
        let testPuzzle = """
        530070000
        600195000
        098000060
        800060003
        400803001
        700020006
        060000280
        000419005
        000080079
        """

        return SudokuGrid(string: testPuzzle.replacingOccurrences(of: "\n", with: ""))!
    }
}

// MARK: - Test Puzzle Generator

extension MockDigitClassifier {
    /// Generate a known solvable test puzzle for development
    static func testPuzzle() -> SudokuGrid {
        // Classic "world's hardest sudoku" by Arto Inkala
        let puzzle = """
        800000000
        003600000
        070090200
        050007000
        000045700
        000100030
        001000068
        008500010
        090000400
        """
        return SudokuGrid(string: puzzle.replacingOccurrences(of: "\n", with: ""))!
    }

    /// Generate a random valid puzzle for testing
    static func randomTestPuzzle() -> SudokuGrid {
        let puzzles = [
            "530070000600195000098000060800060003400803001700020006060000280000419005000080079",
            "003020600900305001001806400008102900700000008006708200002609500800203009005010300",
            "200080300060070084030500209000105408000000000402706000301007040720040060004010003",
            "000000907000420180000705026100904000050000040000507009920108000034059000507000000",
        ]
        return SudokuGrid(string: puzzles.randomElement()!)!
    }
}
