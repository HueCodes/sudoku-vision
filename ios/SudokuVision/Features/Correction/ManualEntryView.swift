import SwiftUI

/// View for manually entering a sudoku puzzle
struct ManualEntryView: View {
    @Environment(\.dismiss) private var dismiss
    @State private var grid = SudokuGrid()
    @State private var selectedCell: (row: Int, col: Int)?
    @State private var solution: SudokuGrid?
    @State private var isInvalid = false
    @State private var isSolving = false

    private let solver = SudokuSolver()

    var body: some View {
        NavigationStack {
            VStack(spacing: 20) {
                // Sudoku grid
                sudokuGridView
                    .aspectRatio(1, contentMode: .fit)
                    .padding()

                // Number pad
                numberPadView
                    .padding(.horizontal)

                Spacer()

                // Action buttons
                HStack(spacing: 16) {
                    Button("Clear") {
                        grid = SudokuGrid()
                        solution = nil
                        isInvalid = false
                    }
                    .buttonStyle(.bordered)

                    Button("Solve") {
                        Task {
                            await solveGrid()
                        }
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(grid.isEmpty || isSolving)
                }
                .padding(.bottom)
            }
            .navigationTitle("Manual Entry")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button("Cancel") {
                        dismiss()
                    }
                }
            }
            .alert("Invalid Puzzle", isPresented: $isInvalid) {
                Button("OK", role: .cancel) {}
            } message: {
                Text("The puzzle contains conflicts or has no solution.")
            }
        }
    }

    // MARK: - Subviews

    private var sudokuGridView: some View {
        VStack(spacing: 0) {
            ForEach(0..<9, id: \.self) { row in
                HStack(spacing: 0) {
                    ForEach(0..<9, id: \.self) { col in
                        cellView(row: row, col: col)
                    }
                }
            }
        }
        .background(Color.primary.opacity(0.1))
        .clipShape(RoundedRectangle(cornerRadius: 8))
        .overlay(gridLines)
    }

    private func cellView(row: Int, col: Int) -> some View {
        let value = grid[row, col]
        let solutionValue = solution?[row, col]
        let isSelected = selectedCell?.row == row && selectedCell?.col == col
        let isOriginal = value != 0
        let isSolved = !isOriginal && solutionValue != nil && solutionValue != 0

        return Button {
            if solution == nil {
                selectedCell = (row, col)
            }
        } label: {
            ZStack {
                Rectangle()
                    .fill(isSelected ? Color.blue.opacity(0.2) : Color.clear)

                if isOriginal {
                    Text("\(value)")
                        .font(.title2.weight(.bold))
                        .foregroundStyle(.primary)
                } else if isSolved {
                    Text("\(solutionValue!)")
                        .font(.title2.weight(.medium))
                        .foregroundStyle(.blue)
                }
            }
        }
        .buttonStyle(.plain)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .aspectRatio(1, contentMode: .fit)
    }

    private var gridLines: some View {
        GeometryReader { geometry in
            let cellSize = geometry.size.width / 9

            // Thin lines for cells
            Path { path in
                for i in 1..<9 {
                    if i % 3 != 0 {
                        let x = CGFloat(i) * cellSize
                        path.move(to: CGPoint(x: x, y: 0))
                        path.addLine(to: CGPoint(x: x, y: geometry.size.height))

                        let y = CGFloat(i) * cellSize
                        path.move(to: CGPoint(x: 0, y: y))
                        path.addLine(to: CGPoint(x: geometry.size.width, y: y))
                    }
                }
            }
            .stroke(Color.primary.opacity(0.2), lineWidth: 1)

            // Thick lines for 3x3 boxes
            Path { path in
                for i in [3, 6] {
                    let x = CGFloat(i) * cellSize
                    path.move(to: CGPoint(x: x, y: 0))
                    path.addLine(to: CGPoint(x: x, y: geometry.size.height))

                    let y = CGFloat(i) * cellSize
                    path.move(to: CGPoint(x: 0, y: y))
                    path.addLine(to: CGPoint(x: geometry.size.width, y: y))
                }
            }
            .stroke(Color.primary, lineWidth: 2)

            // Border
            RoundedRectangle(cornerRadius: 8)
                .stroke(Color.primary, lineWidth: 2)
        }
    }

    private var numberPadView: some View {
        VStack(spacing: 8) {
            HStack(spacing: 8) {
                ForEach(1...5, id: \.self) { num in
                    numberButton(num)
                }
            }
            HStack(spacing: 8) {
                ForEach(6...9, id: \.self) { num in
                    numberButton(num)
                }
                clearButton
            }
        }
    }

    private func numberButton(_ number: Int) -> some View {
        Button {
            if let cell = selectedCell, solution == nil {
                if grid.isValidPlacement(row: cell.row, col: cell.col, value: number) || grid[cell.row, cell.col] != 0 {
                    grid[cell.row, cell.col] = number
                }
            }
        } label: {
            Text("\(number)")
                .font(.title2.weight(.semibold))
                .frame(maxWidth: .infinity)
                .frame(height: 50)
                .background(Color.blue.opacity(0.1))
                .clipShape(RoundedRectangle(cornerRadius: 8))
        }
        .buttonStyle(.plain)
        .disabled(solution != nil)
    }

    private var clearButton: some View {
        Button {
            if let cell = selectedCell, solution == nil {
                grid[cell.row, cell.col] = 0
            }
        } label: {
            Image(systemName: "delete.left")
                .font(.title2)
                .frame(maxWidth: .infinity)
                .frame(height: 50)
                .background(Color.red.opacity(0.1))
                .clipShape(RoundedRectangle(cornerRadius: 8))
        }
        .buttonStyle(.plain)
        .disabled(solution != nil)
    }

    // MARK: - Actions

    private func solveGrid() async {
        isSolving = true
        defer { isSolving = false }

        let result = await solver.solve(grid)
        switch result {
        case .solved(let solved):
            withAnimation {
                solution = solved
            }
        case .noSolution, .invalid:
            isInvalid = true
        }
    }
}

#Preview {
    ManualEntryView()
}
