"""
Sudoku prediction validator.

Validates digit predictions against sudoku rules and identifies
conflicts that indicate recognition errors.

Features:
- Row, column, and box constraint checking
- Conflict identification with cell locations
- Confidence-aware conflict analysis
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Set, Optional
import numpy as np


@dataclass
class Conflict:
    """Represents a constraint violation."""
    type: str  # 'row', 'column', 'box'
    digit: int
    cells: List[Tuple[int, int]]  # List of (row, col) in conflict
    description: str


@dataclass
class CellInfo:
    """Information about a single cell."""
    row: int
    col: int
    digit: int  # 0 = empty
    confidence: float = 1.0
    alternatives: List[Tuple[int, float]] = field(default_factory=list)  # (digit, prob) pairs


@dataclass
class ValidationResult:
    """Result of validation."""
    is_valid: bool
    conflicts: List[Conflict]
    cells_in_conflict: Set[Tuple[int, int]]

    # Summary stats
    num_conflicts: int = 0
    num_cells_affected: int = 0

    def __post_init__(self):
        self.num_conflicts = len(self.conflicts)
        self.num_cells_affected = len(self.cells_in_conflict)


def get_box_index(row: int, col: int) -> int:
    """Get 3x3 box index (0-8) for a cell."""
    return (row // 3) * 3 + (col // 3)


def get_box_cells(box_index: int) -> List[Tuple[int, int]]:
    """Get all cells in a box."""
    box_row = (box_index // 3) * 3
    box_col = (box_index % 3) * 3
    cells = []
    for r in range(box_row, box_row + 3):
        for c in range(box_col, box_col + 3):
            cells.append((r, c))
    return cells


def validate_predictions(
    cells: List[CellInfo],
) -> ValidationResult:
    """Validate predictions against sudoku constraints.

    Args:
        cells: List of 81 CellInfo objects in row-major order

    Returns:
        ValidationResult with conflicts and affected cells
    """
    # Build grid
    grid = [[0] * 9 for _ in range(9)]
    cell_map = {}  # (row, col) -> CellInfo

    for cell in cells:
        grid[cell.row][cell.col] = cell.digit
        cell_map[(cell.row, cell.col)] = cell

    conflicts = []
    cells_in_conflict = set()

    # Check rows
    for r in range(9):
        digit_positions = {}  # digit -> list of columns
        for c in range(9):
            digit = grid[r][c]
            if digit > 0:
                if digit not in digit_positions:
                    digit_positions[digit] = []
                digit_positions[digit].append(c)

        for digit, cols in digit_positions.items():
            if len(cols) > 1:
                conflict_cells = [(r, c) for c in cols]
                conflicts.append(Conflict(
                    type='row',
                    digit=digit,
                    cells=conflict_cells,
                    description=f"Row {r+1}: digit {digit} appears at columns {[c+1 for c in cols]}"
                ))
                cells_in_conflict.update(conflict_cells)

    # Check columns
    for c in range(9):
        digit_positions = {}
        for r in range(9):
            digit = grid[r][c]
            if digit > 0:
                if digit not in digit_positions:
                    digit_positions[digit] = []
                digit_positions[digit].append(r)

        for digit, rows in digit_positions.items():
            if len(rows) > 1:
                conflict_cells = [(r, c) for r in rows]
                conflicts.append(Conflict(
                    type='column',
                    digit=digit,
                    cells=conflict_cells,
                    description=f"Column {c+1}: digit {digit} appears at rows {[r+1 for r in rows]}"
                ))
                cells_in_conflict.update(conflict_cells)

    # Check 3x3 boxes
    for box in range(9):
        box_cells = get_box_cells(box)
        digit_positions = {}

        for r, c in box_cells:
            digit = grid[r][c]
            if digit > 0:
                if digit not in digit_positions:
                    digit_positions[digit] = []
                digit_positions[digit].append((r, c))

        for digit, positions in digit_positions.items():
            if len(positions) > 1:
                conflicts.append(Conflict(
                    type='box',
                    digit=digit,
                    cells=positions,
                    description=f"Box {box+1}: digit {digit} appears {len(positions)} times"
                ))
                cells_in_conflict.update(positions)

    return ValidationResult(
        is_valid=len(conflicts) == 0,
        conflicts=conflicts,
        cells_in_conflict=cells_in_conflict,
    )


def find_lowest_confidence_in_conflict(
    cells: List[CellInfo],
    cells_in_conflict: Set[Tuple[int, int]],
) -> Optional[CellInfo]:
    """Find the cell with lowest confidence among conflicting cells."""
    if not cells_in_conflict:
        return None

    lowest = None
    lowest_conf = float('inf')

    for cell in cells:
        if (cell.row, cell.col) in cells_in_conflict:
            if cell.confidence < lowest_conf:
                lowest_conf = cell.confidence
                lowest = cell

    return lowest


def get_conflict_graph(conflicts: List[Conflict]) -> dict:
    """Build a graph of cells connected by conflicts.

    Returns:
        dict mapping (row, col) to set of connected cells
    """
    graph = {}

    for conflict in conflicts:
        cells = conflict.cells
        for cell in cells:
            if cell not in graph:
                graph[cell] = set()
            for other in cells:
                if other != cell:
                    graph[cell].add(other)

    return graph


def rank_cells_by_conflict_involvement(
    cells: List[CellInfo],
    conflicts: List[Conflict],
) -> List[Tuple[CellInfo, int, float]]:
    """Rank cells by how many conflicts they're involved in.

    Returns:
        List of (cell, num_conflicts, confidence) sorted by num_conflicts desc, confidence asc
    """
    conflict_count = {}

    for conflict in conflicts:
        for r, c in conflict.cells:
            if (r, c) not in conflict_count:
                conflict_count[(r, c)] = 0
            conflict_count[(r, c)] += 1

    cell_map = {(c.row, c.col): c for c in cells}

    ranked = []
    for (r, c), count in conflict_count.items():
        cell = cell_map.get((r, c))
        if cell:
            ranked.append((cell, count, cell.confidence))

    # Sort by: most conflicts, then lowest confidence
    ranked.sort(key=lambda x: (-x[1], x[2]))

    return ranked


def get_possible_values(
    grid: List[List[int]],
    row: int,
    col: int,
) -> Set[int]:
    """Get possible values for a cell based on sudoku constraints."""
    if grid[row][col] != 0:
        return set()

    possible = set(range(1, 10))

    # Remove values in same row
    for c in range(9):
        if grid[row][c] > 0:
            possible.discard(grid[row][c])

    # Remove values in same column
    for r in range(9):
        if grid[r][col] > 0:
            possible.discard(grid[r][col])

    # Remove values in same box
    box_row = (row // 3) * 3
    box_col = (col // 3) * 3
    for r in range(box_row, box_row + 3):
        for c in range(box_col, box_col + 3):
            if grid[r][c] > 0:
                possible.discard(grid[r][c])

    return possible


if __name__ == "__main__":
    # Test validation
    print("Testing validator...")

    # Create test cells with a conflict (two 5s in row 0)
    cells = []
    for i in range(81):
        r, c = i // 9, i % 9
        digit = 0
        if r == 0 and c == 0:
            digit = 5
        elif r == 0 and c == 3:
            digit = 5  # Conflict!
        elif r == 0 and c == 1:
            digit = 3
        cells.append(CellInfo(row=r, col=c, digit=digit, confidence=0.9))

    result = validate_predictions(cells)

    print(f"Valid: {result.is_valid}")
    print(f"Conflicts: {result.num_conflicts}")

    for conflict in result.conflicts:
        print(f"  - {conflict.description}")

    print(f"Cells in conflict: {result.cells_in_conflict}")
