import AVFoundation
import SwiftUI

/// View model for camera scanning
@MainActor
@Observable
final class CameraScanViewModel {

    enum ScanState: Equatable {
        case idle
        case detecting
        case gridDetected
        case processing
        case solved
        case error(String)

        var statusMessage: String {
            switch self {
            case .idle:
                return "Point camera at sudoku puzzle"
            case .detecting:
                return "Looking for sudoku grid..."
            case .gridDetected:
                return "Grid detected, hold steady..."
            case .processing:
                return "Recognizing digits..."
            case .solved:
                return "Solved!"
            case .error(let message):
                return message
            }
        }

        var statusIcon: String {
            switch self {
            case .idle:
                return "viewfinder"
            case .detecting:
                return "square.dashed"
            case .gridDetected:
                return "rectangle.on.rectangle"
            case .processing:
                return "eye"
            case .solved:
                return "checkmark.circle.fill"
            case .error:
                return "exclamationmark.triangle.fill"
            }
        }
    }

    private(set) var scanState: ScanState = .idle
    private(set) var hasPermission = false
    private(set) var isTorchOn = false
    private(set) var detectedCorners: DetectedGrid?
    private(set) var recognizedPuzzle: SudokuGrid?
    private(set) var solution: SudokuGrid?
    private(set) var confidences: [[Float]]?
    private(set) var processingTime: TimeInterval = 0

    private let cameraManager = CameraManager()
    private let pipeline = ScanPipeline()

    private var isProcessingFrame = false

    var captureSession: AVCaptureSession {
        cameraManager.captureSession
    }

    func startScanning() async {
        hasPermission = await cameraManager.requestAccess()
        guard hasPermission else { return }

        cameraManager.setFrameHandler(skipInterval: 15) { [weak self] buffer in
            Task { @MainActor in
                await self?.processFrame(buffer)
            }
        }

        do {
            try await cameraManager.startCapture()
            scanState = .detecting
        } catch {
            scanState = .error(error.localizedDescription)
        }
    }

    func stopScanning() {
        cameraManager.stopCapture()
        scanState = .idle
    }

    func toggleTorch() {
        do {
            try cameraManager.toggleTorch()
            isTorchOn.toggle()
        } catch {
            // Ignore torch errors
        }
    }

    func reset() {
        Task {
            await pipeline.resetStability()
        }
        scanState = .detecting
        detectedCorners = nil
        recognizedPuzzle = nil
        solution = nil
        confidences = nil
        isProcessingFrame = false
    }

    // MARK: - Frame Processing

    private var lastGridDetectedTime: Date?

    private var frameCount = 0

    private func processFrame(_ buffer: CVPixelBuffer) async {
        frameCount += 1
        if frameCount == 1 {
            print("[ViewModel] First frame received!")
        }
        if frameCount % 50 == 0 {
            print("[ViewModel] Frame \(frameCount), state=\(scanState), processing=\(isProcessingFrame)")
        }

        // Skip if already solved or processing
        guard scanState != .solved && !isProcessingFrame else { return }

        isProcessingFrame = true
        defer { isProcessingFrame = false }

        // Transfer buffer to pipeline - safe because buffer is used synchronously
        nonisolated(unsafe) let unsafeBuffer = buffer
        let result = await pipeline.processFrame(unsafeBuffer)

        switch result {
        case .success(let pipelineResult):
            detectedCorners = pipelineResult.detectedGrid
            recognizedPuzzle = pipelineResult.recognizedPuzzle
            solution = pipelineResult.solution
            confidences = pipelineResult.confidences
            processingTime = pipelineResult.processingTime
            scanState = .solved
            triggerSuccessHaptic()

        case .failure(let error):
            switch error {
            case .noGridDetected:
                // No grid found - keep detecting
                if detectedCorners != nil {
                    // Grid was lost
                    detectedCorners = nil
                    lastGridDetectedTime = nil
                }
                if scanState != .detecting {
                    scanState = .detecting
                }

            case .perspectiveCorrectionFailed, .cellExtractionFailed, .classificationFailed:
                // Grid detected but processing failed - show "hold steady" state
                if scanState == .detecting {
                    // First grid detection - trigger haptic
                    triggerGridDetectedHaptic()
                    lastGridDetectedTime = Date()
                }
                scanState = .gridDetected

            case .invalidPuzzle:
                // Puzzle has conflicts - likely recognition error
                scanState = .error("Invalid puzzle - try adjusting angle")

            case .noSolution:
                // Recognized but unsolvable
                scanState = .error("Couldn't solve - check lighting")
            }
        }
    }

    private func triggerSuccessHaptic() {
        let generator = UINotificationFeedbackGenerator()
        generator.notificationOccurred(.success)
    }

    private func triggerGridDetectedHaptic() {
        let generator = UIImpactFeedbackGenerator(style: .light)
        generator.impactOccurred()
    }
}
