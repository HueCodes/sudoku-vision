import SwiftUI

/// Displays detected grid corners as an overlay
struct GridOverlayView: View {
    let corners: DetectedGrid

    var body: some View {
        GeometryReader { geometry in
            let points = corners.points(in: geometry.size)

            // Draw grid outline
            Path { path in
                guard points.count == 4 else { return }
                path.move(to: points[0])
                path.addLine(to: points[1])
                path.addLine(to: points[2])
                path.addLine(to: points[3])
                path.closeSubpath()
            }
            .stroke(Color.green, lineWidth: 3)

            // Draw corner indicators
            ForEach(0..<4, id: \.self) { index in
                Circle()
                    .fill(Color.green)
                    .frame(width: 12, height: 12)
                    .position(points[index])
            }
        }
    }
}

#Preview {
    GridOverlayView(
        corners: DetectedGrid(
            topLeft: CGPoint(x: 0.1, y: 0.2),
            topRight: CGPoint(x: 0.9, y: 0.2),
            bottomRight: CGPoint(x: 0.9, y: 0.8),
            bottomLeft: CGPoint(x: 0.1, y: 0.8),
            confidence: 0.95
        )
    )
    .background(Color.black)
}
