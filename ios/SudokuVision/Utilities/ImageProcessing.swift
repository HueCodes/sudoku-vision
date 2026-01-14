import CoreGraphics
import CoreImage
import CoreVideo
import UIKit

/// Image processing utilities for the detection pipeline
enum ImageProcessing {

    /// Convert CVPixelBuffer to CIImage
    static func ciImage(from pixelBuffer: CVPixelBuffer) -> CIImage {
        CIImage(cvPixelBuffer: pixelBuffer)
    }

    /// Convert CIImage to UIImage (for debugging/display)
    static func uiImage(from ciImage: CIImage, context: CIContext? = nil) -> UIImage? {
        let ctx = context ?? CIContext()
        guard let cgImage = ctx.createCGImage(ciImage, from: ciImage.extent) else {
            return nil
        }
        return UIImage(cgImage: cgImage)
    }

    /// Load image from file path
    static func loadImage(at path: String) -> CIImage? {
        guard let uiImage = UIImage(contentsOfFile: path),
              let cgImage = uiImage.cgImage else {
            return nil
        }
        return CIImage(cgImage: cgImage)
    }

    /// Load image from bundle resource
    static func loadImage(named name: String) -> CIImage? {
        guard let uiImage = UIImage(named: name),
              let cgImage = uiImage.cgImage else {
            return nil
        }
        return CIImage(cgImage: cgImage)
    }

    /// Create a CVPixelBuffer from CIImage (for testing)
    static func createPixelBuffer(from ciImage: CIImage, context: CIContext? = nil) -> CVPixelBuffer? {
        let ctx = context ?? CIContext()

        let width = Int(ciImage.extent.width)
        let height = Int(ciImage.extent.height)

        var pixelBuffer: CVPixelBuffer?
        let attributes: [String: Any] = [
            kCVPixelBufferCGImageCompatibilityKey as String: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey as String: true
        ]

        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            width,
            height,
            kCVPixelFormatType_32BGRA,
            attributes as CFDictionary,
            &pixelBuffer
        )

        guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
            return nil
        }

        ctx.render(ciImage, to: buffer)
        return buffer
    }

    /// Convert extracted cell pixels to grayscale UIImage (for debugging)
    static func uiImage(from pixels: [Float], size: Int = 28) -> UIImage? {
        guard pixels.count == size * size else { return nil }

        // Convert to UInt8
        let bytes = pixels.map { UInt8(max(0, min(255, $0 * 255))) }

        // Create grayscale image
        let colorSpace = CGColorSpaceCreateDeviceGray()
        guard let provider = CGDataProvider(data: Data(bytes) as CFData),
              let cgImage = CGImage(
                width: size,
                height: size,
                bitsPerComponent: 8,
                bitsPerPixel: 8,
                bytesPerRow: size,
                space: colorSpace,
                bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
                provider: provider,
                decode: nil,
                shouldInterpolate: false,
                intent: .defaultIntent
              ) else {
            return nil
        }

        return UIImage(cgImage: cgImage)
    }

    /// Apply adaptive thresholding approximation using Core Image
    static func adaptiveThreshold(_ image: CIImage, blockSize: Int = 11) -> CIImage {
        // Core Image doesn't have true adaptive threshold, so we approximate
        // using local contrast enhancement

        var result = image

        // Apply unsharp mask for local contrast
        if let unsharp = CIFilter(name: "CIUnsharpMask") {
            unsharp.setValue(result, forKey: kCIInputImageKey)
            unsharp.setValue(2.5, forKey: kCIInputRadiusKey)
            unsharp.setValue(1.0, forKey: kCIInputIntensityKey)
            result = unsharp.outputImage ?? result
        }

        // Apply threshold using color matrix
        if let threshold = CIFilter(name: "CIColorMatrix") {
            threshold.setValue(result, forKey: kCIInputImageKey)
            // High contrast to simulate threshold
            threshold.setValue(CIVector(x: 2, y: 0, z: 0, w: 0), forKey: "inputRVector")
            threshold.setValue(CIVector(x: 0, y: 2, z: 0, w: 0), forKey: "inputGVector")
            threshold.setValue(CIVector(x: 0, y: 0, z: 2, w: 0), forKey: "inputBVector")
            threshold.setValue(CIVector(x: -0.5, y: -0.5, z: -0.5, w: 1), forKey: "inputBiasVector")
            result = threshold.outputImage ?? result
        }

        return result
    }
}
