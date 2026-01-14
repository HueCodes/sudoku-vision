import CoreImage
import CoreVideo

/// Corrects perspective distortion to produce a square grid image
final class PerspectiveCorrector {
    private let context: CIContext

    init() {
        // Use Metal for GPU acceleration
        context = CIContext(options: [.useSoftwareRenderer: false])
    }

    /// Apply perspective correction to extract a square grid from detected corners.
    /// - Parameters:
    ///   - pixelBuffer: Source camera frame
    ///   - corners: Detected grid corners in normalized coordinates (0-1)
    ///   - outputSize: Size of the output square image (default 450x450)
    /// - Returns: Perspective-corrected CIImage of the grid, or nil on failure
    func correctPerspective(
        pixelBuffer: CVPixelBuffer,
        corners: DetectedGrid,
        outputSize: CGSize = CGSize(width: 450, height: 450)
    ) -> CIImage? {
        let sourceImage = CIImage(cvPixelBuffer: pixelBuffer)
        let imageSize = sourceImage.extent.size

        // Convert normalized coordinates to pixel coordinates
        // Note: CIImage origin is bottom-left, DetectedGrid uses top-left origin
        let topLeft = CIVector(
            x: corners.topLeft.x * imageSize.width,
            y: (1 - corners.topLeft.y) * imageSize.height
        )
        let topRight = CIVector(
            x: corners.topRight.x * imageSize.width,
            y: (1 - corners.topRight.y) * imageSize.height
        )
        let bottomRight = CIVector(
            x: corners.bottomRight.x * imageSize.width,
            y: (1 - corners.bottomRight.y) * imageSize.height
        )
        let bottomLeft = CIVector(
            x: corners.bottomLeft.x * imageSize.width,
            y: (1 - corners.bottomLeft.y) * imageSize.height
        )

        // Apply perspective correction filter
        guard let filter = CIFilter(name: "CIPerspectiveCorrection") else {
            return nil
        }

        filter.setValue(sourceImage, forKey: kCIInputImageKey)
        filter.setValue(topLeft, forKey: "inputTopLeft")
        filter.setValue(topRight, forKey: "inputTopRight")
        filter.setValue(bottomRight, forKey: "inputBottomRight")
        filter.setValue(bottomLeft, forKey: "inputBottomLeft")

        guard let corrected = filter.outputImage else {
            return nil
        }

        // Scale to desired output size
        let scaleX = outputSize.width / corrected.extent.width
        let scaleY = outputSize.height / corrected.extent.height
        let scaled = corrected.transformed(by: CGAffineTransform(scaleX: scaleX, y: scaleY))

        // Translate to origin
        let translated = scaled.transformed(
            by: CGAffineTransform(translationX: -scaled.extent.origin.x, y: -scaled.extent.origin.y)
        )

        return translated
    }

    /// Render CIImage to CGImage for further processing
    func render(_ image: CIImage) -> CGImage? {
        context.createCGImage(image, from: image.extent)
    }

    /// Apply preprocessing for better digit recognition:
    /// grayscale, contrast enhancement, and adaptive threshold approximation
    func preprocess(_ image: CIImage) -> CIImage {
        var result = image

        // Convert to grayscale
        if let grayscale = CIFilter(name: "CIPhotoEffectMono") {
            grayscale.setValue(result, forKey: kCIInputImageKey)
            result = grayscale.outputImage ?? result
        }

        // Increase contrast
        if let contrast = CIFilter(name: "CIColorControls") {
            contrast.setValue(result, forKey: kCIInputImageKey)
            contrast.setValue(1.5, forKey: kCIInputContrastKey)
            contrast.setValue(0.0, forKey: kCIInputSaturationKey)
            contrast.setValue(0.1, forKey: kCIInputBrightnessKey)
            result = contrast.outputImage ?? result
        }

        // Sharpen edges
        if let sharpen = CIFilter(name: "CISharpenLuminance") {
            sharpen.setValue(result, forKey: kCIInputImageKey)
            sharpen.setValue(0.5, forKey: kCIInputSharpnessKey)
            result = sharpen.outputImage ?? result
        }

        return result
    }
}
