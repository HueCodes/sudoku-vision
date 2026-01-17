import CoreImage
import CoreVideo
import Foundation

/// Pipeline state machine
enum PipelineState: Equatable, Sendable {
    case idle
    case scanning
    case gridDetected(DetectedGrid)
    case extractingCells
    case classifyingDigits
    case solving
    case solved(original: SudokuGrid, solution: SudokuGrid)
    case error(PipelineError)

    var isProcessing: Bool {
        switch self {
        case .scanning, .gridDetected, .extractingCells, .classifyingDigits, .solving:
            return true
        default:
            return false
        }
    }
}

/// Pipeline errors
enum PipelineError: Error, Equatable, Sendable {
    case noGridDetected
    case perspectiveCorrectionFailed
    case cellExtractionFailed
    case classificationFailed
    case invalidPuzzle
    case noSolution

    var message: String {
        switch self {
        case .noGridDetected:
            return "No sudoku grid detected"
        case .perspectiveCorrectionFailed:
            return "Could not correct perspective"
        case .cellExtractionFailed:
            return "Could not extract cells"
        case .classificationFailed:
            return "Could not read digits"
        case .invalidPuzzle:
            return "Invalid puzzle detected"
        case .noSolution:
            return "No solution exists"
        }
    }
}

/// Result of a complete pipeline run
struct PipelineResult: Sendable {
    let detectedGrid: DetectedGrid
    let recognizedPuzzle: SudokuGrid
    let solution: SudokuGrid
    let confidences: [[Float]]
    let processingTime: TimeInterval
}

/// Orchestrates the full detection and solving pipeline
actor ScanPipeline {

    private let gridDetector: GridDetector
    private let perspectiveCorrector: PerspectiveCorrector
    private let cellExtractor: CellExtractor
    private let digitClassifier: any DigitClassifierProtocol
    private let solver: SudokuSolver

    // Stability tracking
    private var lastRecognizedGrid: SudokuGrid?
    private var stableFrameCount = 0
    private let requiredStableFrames = 3

    // Configuration
    private let confidenceThreshold: Float = 0.7
    private let gridOutputSize = CGSize(width: 450, height: 450)

    /// Initialize the pipeline with CoreML classifier.
    /// Falls back to mock classifier if CoreML fails (e.g., on simulator).
    init() {
        self.gridDetector = GridDetector()
        self.perspectiveCorrector = PerspectiveCorrector()
        self.cellExtractor = CellExtractor()
        self.solver = SudokuSolver()

        // Try to initialize CoreML classifier
        // Fall back to mock for simulator or if model is missing
        do {
            self.digitClassifier = try CoreMLDigitClassifier()
            print("[ScanPipeline] Using CoreML digit classifier")
        } catch {
            print("[ScanPipeline] CoreML unavailable: \(error.localizedDescription)")
            print("[ScanPipeline] Falling back to mock classifier")
            self.digitClassifier = MockDigitClassifier()
        }
    }

    /// Initialize with a specific classifier (for testing).
    init(classifier: any DigitClassifierProtocol) {
        self.gridDetector = GridDetector()
        self.perspectiveCorrector = PerspectiveCorrector()
        self.cellExtractor = CellExtractor()
        self.solver = SudokuSolver()
        self.digitClassifier = classifier
    }

    /// Process a single camera frame through the full pipeline
    func processFrame(_ buffer: sending CVPixelBuffer) async -> Result<PipelineResult, PipelineError> {
        let startTime = CFAbsoluteTimeGetCurrent()

        // Step 1: Detect grid
        guard let detectedGrid = gridDetector.detectGrid(in: buffer) else {
            resetStability()
            // Log occasionally
            if Int.random(in: 0..<20) == 0 {
                print("[Pipeline] No grid detected")
            }
            return .failure(.noGridDetected)
        }
        print("[Pipeline] Grid found! conf=\(detectedGrid.confidence)")

        // Step 2: Perspective correction
        guard let warpedImage = perspectiveCorrector.correctPerspective(
            pixelBuffer: buffer,
            corners: detectedGrid,
            outputSize: gridOutputSize
        ) else {
            print("[Pipeline] Perspective correction failed")
            return .failure(.perspectiveCorrectionFailed)
        }

        // Step 3: Preprocess and extract cells
        let preprocessed = perspectiveCorrector.preprocess(warpedImage)
        let extractedCells = cellExtractor.extractCells(from: preprocessed)

        guard extractedCells.cells.count == 81 else {
            print("[Pipeline] Cell extraction failed - got \(extractedCells.cells.count)")
            return .failure(.cellExtractionFailed)
        }

        // Step 4: Classify digits
        let classification = await digitClassifier.classify(cells: extractedCells)
        let recognizedPuzzle = classification.grid

        // Step 5: Check stability (require consistent recognition)
        if !checkStability(recognizedPuzzle) {
            // Only print occasionally
            if stableFrameCount == 1 {
                // Count non-empty cells
                var filledCount = 0
                for row in 0..<9 {
                    for col in 0..<9 {
                        if recognizedPuzzle[row, col] != 0 {
                            filledCount += 1
                        }
                    }
                }
                print("[Pipeline] Stabilizing... conf=\(classification.averageConfidence), filled=\(filledCount)/81")
            }
            return .failure(.noGridDetected) // Not stable yet
        }
        print("[Pipeline] Stable! Solving...")

        // Step 6: Solve
        let solveResult = await solver.solve(recognizedPuzzle)

        let processingTime = CFAbsoluteTimeGetCurrent() - startTime

        switch solveResult {
        case .solved(let solution):
            return .success(PipelineResult(
                detectedGrid: detectedGrid,
                recognizedPuzzle: recognizedPuzzle,
                solution: solution,
                confidences: classification.confidences,
                processingTime: processingTime
            ))
        case .noSolution:
            return .failure(.noSolution)
        case .invalid:
            return .failure(.invalidPuzzle)
        }
    }

    /// Process a static image (for testing)
    func processImage(_ image: CIImage, withGrid grid: DetectedGrid) async -> Result<PipelineResult, PipelineError> {
        let startTime = CFAbsoluteTimeGetCurrent()

        // Create pixel buffer from image
        guard let buffer = ImageProcessing.createPixelBuffer(from: image) else {
            return .failure(.perspectiveCorrectionFailed)
        }

        // Step 2: Perspective correction
        guard let warpedImage = perspectiveCorrector.correctPerspective(
            pixelBuffer: buffer,
            corners: grid,
            outputSize: gridOutputSize
        ) else {
            return .failure(.perspectiveCorrectionFailed)
        }

        // Step 3: Preprocess and extract cells
        let preprocessed = perspectiveCorrector.preprocess(warpedImage)
        let extractedCells = cellExtractor.extractCells(from: preprocessed)

        guard extractedCells.cells.count == 81 else {
            return .failure(.cellExtractionFailed)
        }

        // Step 4: Classify digits
        let classification = await digitClassifier.classify(cells: extractedCells)
        let recognizedPuzzle = classification.grid

        // Step 5: Solve (skip stability check for static images)
        let solveResult = await solver.solve(recognizedPuzzle)

        let processingTime = CFAbsoluteTimeGetCurrent() - startTime

        switch solveResult {
        case .solved(let solution):
            return .success(PipelineResult(
                detectedGrid: grid,
                recognizedPuzzle: recognizedPuzzle,
                solution: solution,
                confidences: classification.confidences,
                processingTime: processingTime
            ))
        case .noSolution:
            return .failure(.noSolution)
        case .invalid:
            return .failure(.invalidPuzzle)
        }
    }

    /// Reset stability tracking
    func resetStability() {
        lastRecognizedGrid = nil
        stableFrameCount = 0
    }

    /// Check if recognition is stable
    private func checkStability(_ grid: SudokuGrid) -> Bool {
        if grid == lastRecognizedGrid {
            stableFrameCount += 1
        } else {
            lastRecognizedGrid = grid
            stableFrameCount = 1
        }
        return stableFrameCount >= requiredStableFrames
    }
}
