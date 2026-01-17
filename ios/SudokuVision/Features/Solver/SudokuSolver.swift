import Foundation

/// Result of a sudoku solve attempt
enum SolveResult: Sendable, Equatable {
    case solved(SudokuGrid)
    case noSolution
    case invalid
}

/// Thread-safe wrapper around the C sudoku solver.
/// Uses an actor to ensure single-threaded access since C code may use global state.
actor SudokuSolver {

    /// Solve a sudoku puzzle.
    /// Returns the solution if found, or an error state otherwise.
    func solve(_ puzzle: SudokuGrid) -> SolveResult {
        // Create contiguous storage for 9x9 grid
        var storage = [Int32](repeating: 0, count: 81)

        // Copy puzzle into flat array (row-major order)
        for row in 0..<9 {
            for col in 0..<9 {
                storage[row * 9 + col] = Int32(puzzle[row, col])
            }
        }

        // Call C solver
        let resultCode = storage.withUnsafeMutableBufferPointer { buffer -> Int32 in
            buffer.baseAddress!.withMemoryRebound(
                to: CRow.self,
                capacity: 9
            ) { gridPtr in
                solve_sudoku(gridPtr)
            }
        }

        switch resultCode {
        case 1: // SOLVE_SUCCESS
            var solution = SudokuGrid()
            for row in 0..<9 {
                for col in 0..<9 {
                    solution[row, col] = Int(storage[row * 9 + col])
                }
            }
            return .solved(solution)
        case 0: // SOLVE_NOSOLUTION
            return .noSolution
        default: // SOLVE_INVALID (-1)
            return .invalid
        }
    }

    /// Validate a puzzle without solving it
    func validate(_ puzzle: SudokuGrid) -> Bool {
        var storage = [Int32](repeating: 0, count: 81)

        for row in 0..<9 {
            for col in 0..<9 {
                storage[row * 9 + col] = Int32(puzzle[row, col])
            }
        }

        return storage.withUnsafeMutableBufferPointer { buffer -> Bool in
            buffer.baseAddress!.withMemoryRebound(
                to: CRow.self,
                capacity: 9
            ) { gridPtr in
                validate_grid(gridPtr) != 0
            }
        }
    }
}

// MARK: - C Type Alias

/// C expects `int grid[9][9]` which is a contiguous block of 81 Int32s.
/// We use a tuple of 9 rows, each being a tuple of 9 Int32s.
private typealias CRow = (Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32)
private typealias CGrid = (CRow, CRow, CRow, CRow, CRow, CRow, CRow, CRow, CRow)
