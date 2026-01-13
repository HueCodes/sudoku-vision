#ifndef SUDOKU_H
#define SUDOKU_H

// Size of the Sudoku grid (9x9)
#define N 9

// Value used for empty cells
#define UNASSIGNED 0

// Function declarations
int solve_sudoku(int grid[N][N]);
int find_unassigned_location(int grid[N][N], int* row, int* col);
int is_safe(int grid[N][N], int row, int col, int num);
int used_in_row(int grid[N][N], int row, int num);
int used_in_col(int grid[N][N], int col, int num);
int used_in_box(int grid[N][N], int box_start_row, int box_start_col, int num);
void print_grid(int grid[N][N]);
void load_example_puzzle(int grid[N][N]);
void input_puzzle(int grid[N][N]);
void input_puzzle_line_by_line(int grid[N][N]);

#endif // SUDOKU_H