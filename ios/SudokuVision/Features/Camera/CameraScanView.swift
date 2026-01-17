import SwiftUI

/// Main camera scanning view with overlay
struct CameraScanView: View {
    @State private var viewModel = CameraScanViewModel()

    var body: some View {
        ZStack {
            // Camera preview
            if viewModel.hasPermission {
                CameraPreviewView(session: viewModel.captureSession)
                    .ignoresSafeArea()

                // Grid overlay when detected
                if let corners = viewModel.detectedCorners {
                    GridOverlayView(corners: corners)
                }

                // Solution overlay when solved
                if let solution = viewModel.solution,
                   let original = viewModel.recognizedPuzzle,
                   let corners = viewModel.detectedCorners {
                    SolutionOverlayView(
                        corners: corners,
                        original: original,
                        solution: solution
                    )
                }

                // Status indicator and controls
                VStack {
                    Spacer()

                    if viewModel.scanState == .solved {
                        // Show processing time and reset button
                        VStack(spacing: 12) {
                            Text(String(format: "Solved in %.0f ms", viewModel.processingTime * 1000))
                                .font(.caption)
                                .foregroundStyle(.secondary)

                            statusView

                            Button("Scan Another") {
                                viewModel.reset()
                            }
                            .buttonStyle(.borderedProminent)
                        }
                        .padding()
                    } else {
                        statusView
                            .padding()
                    }
                }
            } else {
                permissionDeniedView
            }
        }
        .task {
            await viewModel.startScanning()
        }
        .onDisappear {
            viewModel.stopScanning()
        }
        .toolbar {
            ToolbarItem(placement: .topBarLeading) {
                if viewModel.hasPermission {
                    Button {
                        viewModel.toggleTorch()
                    } label: {
                        Image(systemName: viewModel.isTorchOn ? "flashlight.on.fill" : "flashlight.off.fill")
                    }
                }
            }
        }
    }

    @ViewBuilder
    private var statusView: some View {
        let state = viewModel.scanState

        Group {
            switch state {
            case .solved:
                statusLabel(state.statusMessage, icon: state.statusIcon)
                    .foregroundStyle(.green)
            case .gridDetected:
                statusLabel(state.statusMessage, icon: state.statusIcon)
                    .foregroundStyle(.blue)
            case .error:
                statusLabel(state.statusMessage, icon: state.statusIcon)
                    .foregroundStyle(.red)
            default:
                statusLabel(state.statusMessage, icon: state.statusIcon)
            }
        }
    }

    private func statusLabel(_ text: String, icon: String) -> some View {
        HStack {
            Image(systemName: icon)
            Text(text)
        }
        .font(.subheadline.weight(.medium))
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
        .background(.ultraThinMaterial)
        .clipShape(Capsule())
    }

    private var permissionDeniedView: some View {
        VStack(spacing: 20) {
            Image(systemName: "camera.fill")
                .font(.system(size: 60))
                .foregroundStyle(.secondary)

            Text("Camera Access Required")
                .font(.title2.weight(.semibold))

            Text("Sudoku Vision needs camera access to scan puzzles.")
                .font(.body)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal)

            Button("Open Settings") {
                if let url = URL(string: UIApplication.openSettingsURLString) {
                    UIApplication.shared.open(url)
                }
            }
            .buttonStyle(.borderedProminent)
        }
        .padding()
    }
}

#Preview {
    NavigationStack {
        CameraScanView()
    }
}
