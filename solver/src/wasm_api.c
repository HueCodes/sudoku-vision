#include <emscripten/emscripten.h>
#include "../include/sudoku.h"

// Grid storage for WASM interface
static int grid[N][N];

// Export: Set a cell value (called from JS to build the puzzle)
EMSCRIPTEN_KEEPALIVE
void set_cell(int row, int col, int value) {
    if (row >= 0 && row < N && col >= 0 && col < N) {
        grid[row][col] = value;
    }
}

// Export: Get a cell value (called from JS to read solution)
EMSCRIPTEN_KEEPALIVE
int get_cell(int row, int col) {
    if (row >= 0 && row < N && col >= 0 && col < N) {
        return grid[row][col];
    }
    return -1;
}

// Export: Clear the grid
EMSCRIPTEN_KEEPALIVE
void clear_grid(void) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            grid[i][j] = UNASSIGNED;
        }
    }
}

// Export: Solve the puzzle
// Returns:
//   SOLVE_SUCCESS (1)    - Solved successfully
//   SOLVE_NOSOLUTION (0) - Valid input but no solution
//   SOLVE_INVALID (-1)   - Invalid input (duplicates/out-of-range)
EMSCRIPTEN_KEEPALIVE
int solve(void) {
    return solve_sudoku(grid);
}

// Export: Validate current grid state (no conflicts, values in range)
// Returns 1 if valid, 0 if invalid
EMSCRIPTEN_KEEPALIVE
int is_valid(void) {
    return validate_grid(grid);
}
