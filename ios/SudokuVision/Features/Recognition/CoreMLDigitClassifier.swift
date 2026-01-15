import CoreML
import Foundation

/// Production digit classifier using CoreML inference.
///
/// This classifier uses a CNN trained on MNIST to recognize digits 1-9 in sudoku cells.
/// Empty cells are detected via pixel variance analysis before running inference.
///
/// ## Architecture
/// - Input: 28x28 grayscale pixels, normalized [0,1], MNIST-style (white on black)
/// - Output: 10-class logits (indices 0-9)
/// - For sudoku, we only use predictions for classes 1-9 (digit 0 doesn't exist in sudoku)
///
/// ## Usage
/// ```swift
/// let classifier = try CoreMLDigitClassifier()
/// let result = await classifier.classify(cells: extractedCells)
/// ```
actor CoreMLDigitClassifier: DigitClassifierProtocol {

    // MARK: - Properties

    /// The compiled CoreML model
    private let model: MLModel

    /// Thresholds for empty cell detection
    private let emptyMeanThreshold: Float = 0.12
    private let emptyVarianceThreshold: Float = 0.02

    /// Minimum confidence to accept a prediction
    private let minConfidenceThreshold: Float = 0.5

    // MARK: - Initialization

    /// Initialize with the bundled CoreML model.
    ///
    /// - Throws: CoreML errors if model loading fails
    init() throws {
        let config = MLModelConfiguration()
        config.computeUnits = .all  // Use Neural Engine when available

        // Load the Xcode-generated model class
        // Xcode auto-generates "DigitClassifier" class from DigitClassifier.mlpackage
        let digitClassifierModel = try DigitClassifier(configuration: config)
        self.model = digitClassifierModel.model
    }

    /// Initialize with a custom model (for testing).
    ///
    /// - Parameter model: A pre-loaded MLModel instance
    init(model: MLModel) {
        self.model = model
    }

    // MARK: - DigitClassifierProtocol

    /// Classify all 81 cells from a sudoku grid.
    ///
    /// This method processes each cell sequentially, first checking if it's empty
    /// using pixel analysis, then running CoreML inference for non-empty cells.
    ///
    /// - Parameter cells: Extracted cell data from the grid
    /// - Returns: Classification results with predictions and confidences
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

    /// Classify a single cell.
    ///
    /// - Parameter cell: Extracted cell with 28x28 normalized pixels
    /// - Returns: Prediction with digit (0-9) and confidence
    func classifyCell(_ cell: ExtractedCell) async -> DigitPrediction {
        // Step 1: Check if cell is empty using pixel analysis
        // This is faster than running inference and more reliable for empty detection
        if isEmptyCell(cell) {
            return DigitPrediction(digit: 0, confidence: 0.95)
        }

        // Step 2: Run CoreML inference for non-empty cells
        return runInference(on: cell)
    }

    // MARK: - Empty Cell Detection

    /// Determine if a cell is empty based on pixel statistics.
    ///
    /// Empty cells have:
    /// - Low mean intensity (mostly black after MNIST-style inversion)
    /// - Low variance (uniform pixel values)
    ///
    /// - Parameter cell: The cell to analyze
    /// - Returns: True if the cell appears to be empty
    private func isEmptyCell(_ cell: ExtractedCell) -> Bool {
        let pixels = cell.pixels
        let count = Float(pixels.count)

        // Calculate mean intensity
        let mean = pixels.reduce(0, +) / count

        // Calculate variance
        let variance = pixels.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / count

        // Empty cells have low mean (dark) and low variance (uniform)
        return mean < emptyMeanThreshold || variance < emptyVarianceThreshold
    }

    // MARK: - CoreML Inference

    /// Run the CoreML model on a cell's pixel data.
    ///
    /// - Parameter cell: Cell with 28x28 normalized pixels
    /// - Returns: Digit prediction with confidence
    private func runInference(on cell: ExtractedCell) -> DigitPrediction {
        // Convert pixel array to MLMultiArray
        guard let inputArray = createMLMultiArray(from: cell.pixels) else {
            // Fallback to empty if conversion fails
            return DigitPrediction(digit: 0, confidence: 0.0)
        }

        // Create input feature provider
        let inputFeatures = CoreMLDigitClassifierInput(input: inputArray)

        // Run inference
        guard let output = try? model.prediction(from: inputFeatures) else {
            return DigitPrediction(digit: 0, confidence: 0.0)
        }

        // Extract logits from output
        guard let logits = output.featureValue(for: "output")?.multiArrayValue else {
            return DigitPrediction(digit: 0, confidence: 0.0)
        }

        // Convert logits to probabilities and find best prediction
        return extractPrediction(from: logits)
    }

    /// Create MLMultiArray from pixel data.
    ///
    /// - Parameter pixels: Flat array of 784 (28x28) normalized floats
    /// - Returns: MLMultiArray shaped [1, 1, 28, 28] or nil on failure
    private func createMLMultiArray(from pixels: [Float]) -> MLMultiArray? {
        guard pixels.count == 28 * 28 else { return nil }

        // Create array with shape [1, 1, 28, 28] (batch, channels, height, width)
        guard let array = try? MLMultiArray(shape: [1, 1, 28, 28], dataType: .float32) else {
            return nil
        }

        // Copy pixel data
        // The pixels are already in row-major order from CellExtractor
        let pointer = array.dataPointer.assumingMemoryBound(to: Float.self)
        for i in 0..<pixels.count {
            pointer[i] = pixels[i]
        }

        return array
    }

    /// Extract digit prediction from model output logits.
    ///
    /// Applies softmax to logits, then finds the highest probability class.
    /// For sudoku, we prefer digits 1-9 since digit 0 doesn't exist.
    ///
    /// - Parameter logits: Raw model output [1, 10]
    /// - Returns: Best prediction with confidence
    private func extractPrediction(from logits: MLMultiArray) -> DigitPrediction {
        // Extract logits as array
        let count = logits.count
        var logitValues = [Float](repeating: 0, count: count)
        let pointer = logits.dataPointer.assumingMemoryBound(to: Float.self)
        for i in 0..<count {
            logitValues[i] = pointer[i]
        }

        // Apply softmax for probabilities
        let probabilities = softmax(logitValues)

        // Find best prediction among digits 1-9 (sudoku doesn't use 0)
        // We ignore class 0 since it represents the digit "0" in MNIST, not "empty"
        var bestDigit = 1
        var bestProb: Float = probabilities[1]

        for digit in 2...9 {
            if probabilities[digit] > bestProb {
                bestDigit = digit
                bestProb = probabilities[digit]
            }
        }

        // If confidence is too low, return uncertain (digit 0)
        if bestProb < minConfidenceThreshold {
            return DigitPrediction(digit: 0, confidence: bestProb)
        }

        return DigitPrediction(digit: bestDigit, confidence: bestProb)
    }

    /// Compute softmax probabilities from logits.
    ///
    /// - Parameter logits: Raw model outputs
    /// - Returns: Normalized probabilities summing to 1.0
    private func softmax(_ logits: [Float]) -> [Float] {
        // Subtract max for numerical stability
        let maxLogit = logits.max() ?? 0
        let expValues = logits.map { exp($0 - maxLogit) }
        let sumExp = expValues.reduce(0, +)
        return expValues.map { $0 / sumExp }
    }
}

// MARK: - Input Feature Provider

/// Input wrapper for the CoreML model.
///
/// This class provides the input features in the format expected by CoreML.
/// We define it explicitly rather than relying on Xcode-generated classes
/// for more control and clarity.
private class CoreMLDigitClassifierInput: MLFeatureProvider {

    let input: MLMultiArray

    var featureNames: Set<String> {
        return ["input"]
    }

    init(input: MLMultiArray) {
        self.input = input
    }

    func featureValue(for featureName: String) -> MLFeatureValue? {
        if featureName == "input" {
            return MLFeatureValue(multiArray: input)
        }
        return nil
    }
}
