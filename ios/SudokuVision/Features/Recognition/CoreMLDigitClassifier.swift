import CoreML
import Foundation

/// Production digit classifier using CoreML inference.
///
/// This classifier uses a CNN trained on synthetic + real printed digits to recognize
/// digits 1-9 in sudoku cells. The model was trained with CLAHE + adaptive threshold
/// preprocessing which is applied automatically before inference.
///
/// ## Architecture
/// - Input: 28x28 grayscale pixels, preprocessed with CLAHE + adaptive threshold
/// - Normalization: [-1, 1] (not [0, 1])
/// - Output: 10-class logits (0=empty, 1-9=digits)
/// - Class 0 represents empty cells in this model
///
/// ## Usage
/// ```swift
/// let classifier = try CoreMLDigitClassifier()
/// let result = await classifier.classify(cells: extractedCells)
/// ```
actor CoreMLDigitClassifier: DigitClassifierProtocol {

    // MARK: - Properties

    /// The compiled CoreML model (DigitClassifier_v2)
    private let model: MLModel

    /// Thresholds for empty cell detection (pre-inference check)
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

        // Load the v2 model trained on synthetic + real data with CLAHE preprocessing
        // Try v2 first, fall back to v1 if not found
        if let modelURL = Bundle.main.url(forResource: "DigitClassifier_v2", withExtension: "mlmodelc") {
            self.model = try MLModel(contentsOf: modelURL, configuration: config)
        } else if let modelURL = Bundle.main.url(forResource: "DigitClassifier", withExtension: "mlmodelc") {
            self.model = try MLModel(contentsOf: modelURL, configuration: config)
        } else {
            throw NSError(domain: "CoreMLDigitClassifier", code: 1,
                         userInfo: [NSLocalizedDescriptionKey: "Could not find DigitClassifier model in bundle"])
        }
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
    nonisolated(unsafe) private static var cellLogCount = 0

    func classifyCell(_ cell: ExtractedCell) async -> DigitPrediction {
        // Log raw cell stats for first few cells
        Self.cellLogCount += 1
        if Self.cellLogCount <= 3 {
            let rawMean = cell.pixels.reduce(0, +) / Float(cell.pixels.count)
            let rawMin = cell.pixels.min() ?? 0
            let rawMax = cell.pixels.max() ?? 0
            print("[Cell \(Self.cellLogCount)] Raw: mean=\(rawMean), min=\(rawMin), max=\(rawMax), count=\(cell.pixels.count)")
        }

        // Step 1: Apply CLAHE + adaptive threshold preprocessing
        let preprocessedPixels = CellPreprocessor.preprocess(cell.pixels)

        // Log preprocessed stats
        if Self.cellLogCount <= 3 {
            let prepMean = preprocessedPixels.reduce(0, +) / Float(preprocessedPixels.count)
            let prepMin = preprocessedPixels.min() ?? 0
            let prepMax = preprocessedPixels.max() ?? 0
            print("[Cell \(Self.cellLogCount)] Prep: mean=\(prepMean), min=\(prepMin), max=\(prepMax)")
        }

        // Step 2: Run CoreML inference on preprocessed pixels
        return runInference(on: preprocessedPixels)
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

    /// Run the CoreML model on preprocessed pixel data.
    ///
    /// - Parameter pixels: Preprocessed 28x28 pixels in [-1, 1] range
    /// - Returns: Digit prediction with confidence
    nonisolated(unsafe) private static var inferenceLogCount = 0

    private func runInference(on pixels: [Float]) -> DigitPrediction {
        // Convert pixel array to MLMultiArray
        guard let inputArray = createMLMultiArray(from: pixels) else {
            print("[Classifier] Failed to create MLMultiArray")
            return DigitPrediction(digit: 0, confidence: 0.0)
        }

        // Create input feature provider
        let inputFeatures = CoreMLDigitClassifierInput(input: inputArray)

        // Run inference
        guard let output = try? model.prediction(from: inputFeatures) else {
            print("[Classifier] Model prediction failed")
            return DigitPrediction(digit: 0, confidence: 0.0)
        }

        // Log output feature names once
        Self.inferenceLogCount += 1
        if Self.inferenceLogCount == 1 {
            print("[Classifier] Output features: \(output.featureNames)")
        }

        // Extract logits from output - try different possible names
        var logits: MLMultiArray?
        for name in ["output", "var_143", "Identity", "logits"] {
            if let arr = output.featureValue(for: name)?.multiArrayValue {
                logits = arr
                if Self.inferenceLogCount == 1 {
                    print("[Classifier] Using output name: \(name)")
                }
                break
            }
        }

        guard let logits = logits else {
            print("[Classifier] Could not find output tensor")
            return DigitPrediction(digit: 0, confidence: 0.0)
        }

        // Convert logits to probabilities and find best prediction
        let prediction = extractPrediction(from: logits)

        // Log first few predictions
        if Self.inferenceLogCount <= 5 {
            print("[Classifier] Prediction: digit=\(prediction.digit), conf=\(prediction.confidence)")
        }

        return prediction
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
    /// The v2 model uses: class 0 = empty cell, classes 1-9 = digits.
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

        // Find best prediction among all classes (0-9)
        // Class 0 = empty cell, classes 1-9 = digits
        var bestClass = 0
        var bestProb: Float = probabilities[0]

        for classIdx in 1..<probabilities.count {
            if probabilities[classIdx] > bestProb {
                bestClass = classIdx
                bestProb = probabilities[classIdx]
            }
        }

        // If confidence is too low, return uncertain empty (digit 0)
        if bestProb < minConfidenceThreshold {
            return DigitPrediction(digit: 0, confidence: bestProb)
        }

        return DigitPrediction(digit: bestClass, confidence: bestProb)
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
