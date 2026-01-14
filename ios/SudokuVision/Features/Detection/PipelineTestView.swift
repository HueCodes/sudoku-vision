import SwiftUI

/// Test view for running the pipeline on static images (development only)
struct PipelineTestView: View {
    @State private var testResult: TestResult?
    @State private var isProcessing = false
    @State private var selectedImage: String = "sample_1"

    private let pipeline = ScanPipeline()
    private let gridDetector = GridDetector()

    private let testImages = ["sample_1", "sample_2", "sample_3", "sample_4", "sample_5"]

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 20) {
                    // Image picker
                    Picker("Test Image", selection: $selectedImage) {
                        ForEach(testImages, id: \.self) { name in
                            Text(name).tag(name)
                        }
                    }
                    .pickerStyle(.segmented)
                    .padding(.horizontal)

                    // Test image display
                    if let uiImage = UIImage(named: selectedImage) {
                        Image(uiImage: uiImage)
                            .resizable()
                            .aspectRatio(contentMode: .fit)
                            .frame(maxHeight: 300)
                            .clipShape(RoundedRectangle(cornerRadius: 12))
                            .padding(.horizontal)
                    } else {
                        placeholderView
                    }

                    // Run test button
                    Button {
                        Task {
                            await runPipelineTest()
                        }
                    } label: {
                        HStack {
                            if isProcessing {
                                ProgressView()
                                    .tint(.white)
                            }
                            Text(isProcessing ? "Processing..." : "Run Pipeline Test")
                        }
                        .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(isProcessing)
                    .padding(.horizontal)

                    // Results
                    if let result = testResult {
                        resultView(result)
                    }
                }
                .padding(.vertical)
            }
            .navigationTitle("Pipeline Test")
        }
    }

    private var placeholderView: some View {
        VStack {
            Image(systemName: "photo")
                .font(.system(size: 60))
                .foregroundStyle(.secondary)
            Text("Add test images to Assets.xcassets")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .frame(height: 200)
        .frame(maxWidth: .infinity)
        .background(Color.secondary.opacity(0.1))
        .clipShape(RoundedRectangle(cornerRadius: 12))
        .padding(.horizontal)
    }

    @ViewBuilder
    private func resultView(_ result: TestResult) -> some View {
        VStack(alignment: .leading, spacing: 16) {
            // Status
            HStack {
                Image(systemName: result.success ? "checkmark.circle.fill" : "xmark.circle.fill")
                    .foregroundStyle(result.success ? .green : .red)
                Text(result.success ? "Success" : "Failed")
                    .font(.headline)
                Spacer()
                Text(String(format: "%.1f ms", result.processingTime * 1000))
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            if let error = result.error {
                Text("Error: \(error)")
                    .font(.caption)
                    .foregroundStyle(.red)
            }

            if let puzzle = result.recognizedPuzzle {
                VStack(alignment: .leading, spacing: 8) {
                    Text("Recognized Puzzle:")
                        .font(.subheadline.weight(.semibold))

                    Text(puzzle.description)
                        .font(.system(.caption, design: .monospaced))
                        .padding(8)
                        .background(Color.secondary.opacity(0.1))
                        .clipShape(RoundedRectangle(cornerRadius: 8))
                }
            }

            if let solution = result.solution {
                VStack(alignment: .leading, spacing: 8) {
                    Text("Solution:")
                        .font(.subheadline.weight(.semibold))

                    Text(solution.description)
                        .font(.system(.caption, design: .monospaced))
                        .padding(8)
                        .background(Color.green.opacity(0.1))
                        .clipShape(RoundedRectangle(cornerRadius: 8))
                }
            }
        }
        .padding()
        .background(Color.secondary.opacity(0.05))
        .clipShape(RoundedRectangle(cornerRadius: 12))
        .padding(.horizontal)
    }

    private func runPipelineTest() async {
        isProcessing = true
        defer { isProcessing = false }

        let startTime = CFAbsoluteTimeGetCurrent()

        // Load test image
        guard let uiImage = UIImage(named: selectedImage),
              let cgImage = uiImage.cgImage else {
            testResult = TestResult(
                success: false,
                error: "Could not load image",
                processingTime: 0
            )
            return
        }

        let ciImage = CIImage(cgImage: cgImage)

        // Create pixel buffer
        guard let pixelBuffer = ImageProcessing.createPixelBuffer(from: ciImage) else {
            testResult = TestResult(
                success: false,
                error: "Could not create pixel buffer",
                processingTime: CFAbsoluteTimeGetCurrent() - startTime
            )
            return
        }

        // Detect grid
        guard let detectedGrid = await gridDetector.detectGrid(in: pixelBuffer) else {
            testResult = TestResult(
                success: false,
                error: "No grid detected",
                processingTime: CFAbsoluteTimeGetCurrent() - startTime
            )
            return
        }

        // Run pipeline
        let result = await pipeline.processImage(ciImage, withGrid: detectedGrid)
        let processingTime = CFAbsoluteTimeGetCurrent() - startTime

        switch result {
        case .success(let pipelineResult):
            testResult = TestResult(
                success: true,
                recognizedPuzzle: pipelineResult.recognizedPuzzle,
                solution: pipelineResult.solution,
                processingTime: processingTime
            )
        case .failure(let error):
            testResult = TestResult(
                success: false,
                error: error.message,
                processingTime: processingTime
            )
        }
    }
}

// MARK: - Test Result

private struct TestResult {
    let success: Bool
    var error: String?
    var recognizedPuzzle: SudokuGrid?
    var solution: SudokuGrid?
    let processingTime: TimeInterval
}

#Preview {
    PipelineTestView()
}
