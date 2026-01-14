#ifndef SUDOKU_H
#define SUDOKU_H

#include <stdint.h>

// Size of the Sudoku grid (9x9)
#define N 9

// Value used for empty cells
#define UNASSIGNED 0

// Solver return codes
#define SOLVE_SUCCESS    1   // Puzzle solved successfully
#define SOLVE_NOSOLUTION 0   // Valid input but no solution exists
#define SOLVE_INVALID   -1   // Invalid input (duplicates or out-of-range)

// Candidate bitmask: bits 1-9 represent possible values for a cell
// Example: 0b0000001010 means candidates are {2, 4}
typedef uint16_t candidates_t;
#define ALL_CANDIDATES 0x3FE   // bits 1-9 set (0b1111111110)
#define CANDIDATE(n) (1 << (n))  // bit for digit n

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

// Constraint propagation functions
void init_candidates(int grid[N][N], candidates_t cands[N][N]);
int eliminate_from_peers(candidates_t cands[N][N], int row, int col, int num);
int propagate(int grid[N][N], candidates_t cands[N][N]);

// Helper functions
int count_candidates(candidates_t c);
int first_candidate(candidates_t c);
int is_solved(int grid[N][N]);

// Validation
int validate_grid(int grid[N][N]);

// File I/O
int load_from_file(const char* filename, int grid[N][N]);
int save_to_file(const char* filename, int grid[N][N]);

#endif // SUDOKU_H
