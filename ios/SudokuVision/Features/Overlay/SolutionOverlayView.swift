import SwiftUI

/// Overlays solution digits on the camera feed
struct SolutionOverlayView: View {
    let corners: DetectedGrid
    let original: SudokuGrid
    let solution: SudokuGrid

    var body: some View {
        GeometryReader { geometry in
            let points = corners.points(in: geometry.size)
            let transform = perspectiveTransform(from: points, size: geometry.size)

            ForEach(0..<9, id: \.self) { row in
                ForEach(0..<9, id: \.self) { col in
                    let originalValue = original[row, col]
                    let solutionValue = solution[row, col]

                    // Only show solved digits (not original clues)
                    if originalValue == 0 && solutionValue != 0 {
                        let position = cellPosition(row: row, col: col, transform: transform, size: geometry.size)

                        Text("\(solutionValue)")
                            .font(.system(size: cellFontSize(for: geometry.size), weight: .bold, design: .rounded))
                            .foregroundStyle(.blue)
                            .shadow(color: .black.opacity(0.5), radius: 2, x: 1, y: 1)
                            .position(position)
                    }
                }
            }
        }
    }

    /// Calculate cell center position using perspective transform
    private func cellPosition(row: Int, col: Int, transform: CGAffineTransform, size: CGSize) -> CGPoint {
        // Normalized position within grid (0-1)
        let nx = (CGFloat(col) + 0.5) / 9.0
        let ny = (CGFloat(row) + 0.5) / 9.0

        // Apply perspective transform
        let points = corners.points(in: size)
        return interpolatePoint(nx: nx, ny: ny, corners: points)
    }

    /// Bilinear interpolation within quadrilateral
    private func interpolatePoint(nx: CGFloat, ny: CGFloat, corners: [CGPoint]) -> CGPoint {
        let topLeft = corners[0]
        let topRight = corners[1]
        let bottomRight = corners[2]
        let bottomLeft = corners[3]

        // Interpolate along top and bottom edges
        let topPoint = CGPoint(
            x: topLeft.x + (topRight.x - topLeft.x) * nx,
            y: topLeft.y + (topRight.y - topLeft.y) * nx
        )
        let bottomPoint = CGPoint(
            x: bottomLeft.x + (bottomRight.x - bottomLeft.x) * nx,
            y: bottomLeft.y + (bottomRight.y - bottomLeft.y) * nx
        )

        // Interpolate between top and bottom
        return CGPoint(
            x: topPoint.x + (bottomPoint.x - topPoint.x) * ny,
            y: topPoint.y + (bottomPoint.y - topPoint.y) * ny
        )
    }

    private func perspectiveTransform(from points: [CGPoint], size: CGSize) -> CGAffineTransform {
        // Simplified: just use identity for now
        // Full perspective transform would use Core Image CIPerspectiveCorrection
        return .identity
    }

    private func cellFontSize(for size: CGSize) -> CGFloat {
        let gridWidth = abs(corners.topRight.x - corners.topLeft.x) * size.width
        return gridWidth / 9 * 0.6
    }
}

#Preview {
    let original = SudokuGrid(string: "530070000600195000098000060800060003400803001700020006060000280000419005000080079")!
    let solution = SudokuGrid(string: "534678912672195348198342567859761423426853791713924856961537284287419635345286179")!

    return SolutionOverlayView(
        corners: DetectedGrid(
            topLeft: CGPoint(x: 0.1, y: 0.1),
            topRight: CGPoint(x: 0.9, y: 0.1),
            bottomRight: CGPoint(x: 0.9, y: 0.9),
            bottomLeft: CGPoint(x: 0.1, y: 0.9),
            confidence: 0.95
        ),
        original: original,
        solution: solution
    )
    .background(Color.black)
}
