import AVFoundation
import SwiftUI

/// View model for camera scanning
@MainActor
@Observable
final class CameraScanViewModel {

    enum ScanState: Equatable {
        case idle
        case detecting
        case processing
        case solved
        case error(String)

        var statusMessage: String {
            switch self {
            case .idle:
                return "Point camera at sudoku puzzle"
            case .detecting:
                return "Detecting grid..."
            case .processing:
                return "Reading digits..."
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

        cameraManager.setFrameHandler(skipInterval: 3) { [weak self] buffer in
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

    private func processFrame(_ buffer: CVPixelBuffer) async {
        // Skip if already solved or processing
        guard scanState != .solved && !isProcessingFrame else { return }

        isProcessingFrame = true
        defer { isProcessingFrame = false }

        let result = await pipeline.processFrame(buffer)

        switch result {
        case .success(let pipelineResult):
            detectedCorners = pipelineResult.detectedGrid
            recognizedPuzzle = pipelineResult.recognizedPuzzle
            solution = pipelineResult.solution
            confidences = pipelineResult.confidences
            processingTime = pipelineResult.processingTime
            scanState = .solved
            triggerHapticFeedback()

        case .failure(let error):
            switch error {
            case .noGridDetected:
                // Normal state - keep detecting
                detectedCorners = nil
                if scanState != .detecting {
                    scanState = .detecting
                }
            case .perspectiveCorrectionFailed, .cellExtractionFailed, .classificationFailed:
                // Transient errors - keep trying
                scanState = .processing
            case .invalidPuzzle, .noSolution:
                // Show error to user
                scanState = .error(error.message)
            }
        }
    }

    private func triggerHapticFeedback() {
        let generator = UINotificationFeedbackGenerator()
        generator.notificationOccurred(.success)
    }
}
