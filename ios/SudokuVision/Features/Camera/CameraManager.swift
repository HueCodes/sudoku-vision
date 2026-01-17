import AVFoundation
import CoreImage
import Foundation

/// Manages AVCaptureSession for camera input
@MainActor
final class CameraManager: NSObject, ObservableObject {
    @Published private(set) var isRunning = false
    @Published private(set) var error: CameraError?

    let captureSession = AVCaptureSession()
    private let videoOutput = AVCaptureVideoDataOutput()
    private let sessionQueue = DispatchQueue(label: "camera.session")
    private let outputQueue = DispatchQueue(label: "camera.output")

    private var frameHandler: ((CVPixelBuffer) -> Void)?
    // nonisolated(unsafe) because these are only accessed from outputQueue
    nonisolated(unsafe) private var frameSkipCount = 0
    nonisolated(unsafe) private var frameSkipInterval = 3 // Process every Nth frame

    var previewLayer: AVCaptureVideoPreviewLayer {
        let layer = AVCaptureVideoPreviewLayer(session: captureSession)
        layer.videoGravity = .resizeAspectFill
        return layer
    }

    /// Configure frame processing callback
    func setFrameHandler(skipInterval: Int = 3, handler: @escaping (CVPixelBuffer) -> Void) {
        frameSkipInterval = skipInterval
        frameHandler = handler
    }

    /// Request camera permission and configure session
    func requestAccess() async -> Bool {
        let status = AVCaptureDevice.authorizationStatus(for: .video)

        switch status {
        case .authorized:
            return true
        case .notDetermined:
            return await AVCaptureDevice.requestAccess(for: .video)
        case .denied, .restricted:
            error = .permissionDenied
            return false
        @unknown default:
            return false
        }
    }

    /// Configure and start the capture session
    func startCapture() async throws {
        guard await requestAccess() else {
            throw CameraError.permissionDenied
        }

        try await configureSession()

        sessionQueue.async { [weak self] in
            self?.captureSession.startRunning()
            Task { @MainActor in
                self?.isRunning = true
            }
        }
    }

    /// Stop the capture session
    func stopCapture() {
        sessionQueue.async { [weak self] in
            self?.captureSession.stopRunning()
            Task { @MainActor in
                self?.isRunning = false
            }
        }
    }

    /// Toggle torch for low-light conditions
    func toggleTorch() throws {
        guard let device = AVCaptureDevice.default(for: .video),
              device.hasTorch else {
            throw CameraError.torchUnavailable
        }

        try device.lockForConfiguration()
        device.torchMode = device.torchMode == .on ? .off : .on
        device.unlockForConfiguration()
    }

    // MARK: - Private

    private func configureSession() async throws {
        captureSession.beginConfiguration()
        defer { captureSession.commitConfiguration() }

        captureSession.sessionPreset = .hd1280x720  // Lower resolution for better performance

        // Add video input
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) else {
            throw CameraError.deviceNotFound
        }

        do {
            let input = try AVCaptureDeviceInput(device: device)
            if captureSession.canAddInput(input) {
                captureSession.addInput(input)
            } else {
                throw CameraError.configurationFailed
            }
        } catch {
            throw CameraError.configurationFailed
        }

        // Configure video output
        videoOutput.alwaysDiscardsLateVideoFrames = true
        videoOutput.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        videoOutput.setSampleBufferDelegate(self, queue: outputQueue)

        if captureSession.canAddOutput(videoOutput) {
            captureSession.addOutput(videoOutput)
        } else {
            throw CameraError.configurationFailed
        }

        // Set video orientation
        if let connection = videoOutput.connection(with: .video) {
            connection.videoRotationAngle = 90 // Portrait
        }

        // Enable auto-focus
        try? device.lockForConfiguration()
        if device.isFocusModeSupported(.continuousAutoFocus) {
            device.focusMode = .continuousAutoFocus
        }
        if device.isExposureModeSupported(.continuousAutoExposure) {
            device.exposureMode = .continuousAutoExposure
        }
        device.unlockForConfiguration()
    }
}

// MARK: - AVCaptureVideoDataOutputSampleBufferDelegate

extension CameraManager: AVCaptureVideoDataOutputSampleBufferDelegate {
    nonisolated func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        // Frame skipping for battery efficiency
        frameSkipCount += 1
        guard frameSkipCount >= frameSkipInterval else { return }
        frameSkipCount = 0

        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }

        // Call handler on main actor
        nonisolated(unsafe) let unsafeBuffer = pixelBuffer
        Task { @MainActor [weak self] in
            self?.frameHandler?(unsafeBuffer)
        }
    }
}

// MARK: - CameraError

enum CameraError: Error, LocalizedError {
    case permissionDenied
    case deviceNotFound
    case configurationFailed
    case torchUnavailable

    var errorDescription: String? {
        switch self {
        case .permissionDenied:
            return "Camera access denied. Please enable in Settings."
        case .deviceNotFound:
            return "No camera found on this device."
        case .configurationFailed:
            return "Failed to configure camera."
        case .torchUnavailable:
            return "Torch is not available on this device."
        }
    }
}
