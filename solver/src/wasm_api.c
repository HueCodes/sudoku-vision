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

// Export: Solve the puzzle, returns 1 if solved, 0 if unsolvable
EMSCRIPTEN_KEEPALIVE
int solve(void) {
    return solve_sudoku(grid);
}

// Export: Validate current grid state (no conflicts)
EMSCRIPTEN_KEEPALIVE
int is_valid(void) {
    for (int row = 0; row < N; row++) {
        for (int col = 0; col < N; col++) {
            int val = grid[row][col];
            if (val != UNASSIGNED) {
                // Temporarily clear cell to check if value is valid
                grid[row][col] = UNASSIGNED;
                int valid = is_safe(grid, row, col, val);
                grid[row][col] = val;
                if (!valid) return 0;
            }
        }
    }
    return 1;
}
