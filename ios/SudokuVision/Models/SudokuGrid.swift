import Foundation

/// 9x9 Sudoku grid model. Values 0-9 where 0 represents an empty cell.
struct SudokuGrid: Equatable, Sendable, Hashable {
    private var cells: [[Int]]

    init() {
        cells = Array(repeating: Array(repeating: 0, count: 9), count: 9)
    }

    init(cells: [[Int]]) {
        precondition(cells.count == 9 && cells.allSatisfy { $0.count == 9 })
        self.cells = cells
    }

    /// Initialize from an 81-character string (row-major, '.' or '0' for empty)
    init?(string: String) {
        let chars = Array(string)
        guard chars.count == 81 else { return nil }

        var grid: [[Int]] = []
        for row in 0..<9 {
            var rowValues: [Int] = []
            for col in 0..<9 {
                let char = chars[row * 9 + col]
                if char == "." || char == "0" {
                    rowValues.append(0)
                } else if let digit = char.wholeNumberValue, digit >= 1 && digit <= 9 {
                    rowValues.append(digit)
                } else {
                    return nil
                }
            }
            grid.append(rowValues)
        }
        self.cells = grid
    }

    subscript(row: Int, col: Int) -> Int {
        get {
            precondition(row >= 0 && row < 9 && col >= 0 && col < 9)
            return cells[row][col]
        }
        set {
            precondition(row >= 0 && row < 9 && col >= 0 && col < 9)
            precondition(newValue >= 0 && newValue <= 9)
            cells[row][col] = newValue
        }
    }

    var isEmpty: Bool {
        cells.allSatisfy { $0.allSatisfy { $0 == 0 } }
    }

    var filledCount: Int {
        cells.flatMap { $0 }.filter { $0 != 0 }.count
    }

    /// Returns cells that differ from the original (for highlighting solution digits)
    func diff(from original: SudokuGrid) -> [(row: Int, col: Int, value: Int)] {
        var differences: [(row: Int, col: Int, value: Int)] = []
        for row in 0..<9 {
            for col in 0..<9 {
                if self[row, col] != original[row, col] && self[row, col] != 0 {
                    differences.append((row, col, self[row, col]))
                }
            }
        }
        return differences
    }

    /// Check if placing a value at (row, col) would be valid
    func isValidPlacement(row: Int, col: Int, value: Int) -> Bool {
        guard value >= 1 && value <= 9 else { return false }

        // Check row
        for c in 0..<9 where c != col {
            if cells[row][c] == value { return false }
        }

        // Check column
        for r in 0..<9 where r != row {
            if cells[r][col] == value { return false }
        }

        // Check 3x3 box
        let boxRow = (row / 3) * 3
        let boxCol = (col / 3) * 3
        for r in boxRow..<(boxRow + 3) {
            for c in boxCol..<(boxCol + 3) {
                if r != row || c != col {
                    if cells[r][c] == value { return false }
                }
            }
        }

        return true
    }

    /// Convert to 81-character string representation
    func toString() -> String {
        cells.flatMap { $0 }.map { $0 == 0 ? "." : String($0) }.joined()
    }
}

// MARK: - CustomStringConvertible

extension SudokuGrid: CustomStringConvertible {
    var description: String {
        var result = ""
        for row in 0..<9 {
            if row > 0 && row % 3 == 0 {
                result += "------+-------+------\n"
            }
            for col in 0..<9 {
                if col > 0 && col % 3 == 0 {
                    result += " |"
                }
                let value = cells[row][col]
                result += " \(value == 0 ? "." : String(value))"
            }
            result += "\n"
        }
        return result
    }
}
