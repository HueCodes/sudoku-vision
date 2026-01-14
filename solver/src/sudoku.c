#include <stdio.h>
#include <string.h>
#include "../include/sudoku.h"

/* Internal solver with constraint propagation and MRV heuristic */
static int solve_with_candidates(int grid[N][N], candidates_t cands[N][N])
{
    // Propagate constraints
    if (propagate(grid, cands) < 0)
        return 0; // Contradiction

    // Check if solved
    if (is_solved(grid))
        return 1;

    // Find cell with minimum remaining values (MRV heuristic)
    int best_row = -1, best_col = -1;
    int min_count = 10;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (grid[i][j] == UNASSIGNED) {
                int count = count_candidates(cands[i][j]);
                if (count < min_count) {
                    min_count = count;
                    best_row = i;
                    best_col = j;
                }
            }
        }
    }

    if (best_row < 0)
        return 0; // No unassigned cell found but not solved - shouldn't happen

    // Try each candidate for the chosen cell
    candidates_t cell_cands = cands[best_row][best_col];
    for (int num = 1; num <= 9; num++) {
        if (cell_cands & CANDIDATE(num)) {
            // Make copies for backtracking
            int grid_copy[N][N];
            candidates_t cands_copy[N][N];
            memcpy(grid_copy, grid, sizeof(grid_copy));
            memcpy(cands_copy, cands, sizeof(cands_copy));

            // Make tentative assignment
            grid_copy[best_row][best_col] = num;
            cands_copy[best_row][best_col] = 0;
            eliminate_from_peers(cands_copy, best_row, best_col, num);

            // Recurse
            if (solve_with_candidates(grid_copy, cands_copy)) {
                memcpy(grid, grid_copy, sizeof(grid_copy));
                return 1;
            }
        }
    }

    return 0; // Backtrack
}

/*
 * Takes a partially filled-in grid and attempts to assign values to
 * all unassigned locations in such a way to meet the requirements
 * for Sudoku (non-duplication across rows, columns, and boxes).
 * Uses constraint propagation with naked/hidden singles and MRV heuristic.
 *
 * Returns:
 *   SOLVE_SUCCESS (1)    - Puzzle solved successfully
 *   SOLVE_NOSOLUTION (0) - Valid input but no solution exists
 *   SOLVE_INVALID (-1)   - Invalid input (duplicates or out-of-range)
 */
int solve_sudoku(int grid[N][N])
{
    // Validate input first
    if (!validate_grid(grid))
        return SOLVE_INVALID;

    candidates_t cands[N][N];
    init_candidates(grid, cands);
    return solve_with_candidates(grid, cands) ? SOLVE_SUCCESS : SOLVE_NOSOLUTION;
}

/* 
 * Searches the grid to find an entry that is still unassigned. If
 * found, the reference parameters row, col will be set the location
 * that is unassigned, and true is returned. If no unassigned entries
 * remain, false is returned.
 */
int find_unassigned_location(int grid[N][N], int* row, int* col)
{
    for (*row = 0; *row < N; (*row)++)
        for (*col = 0; *col < N; (*col)++)
            if (grid[*row][*col] == UNASSIGNED)
                return 1;
    return 0;
}

/* 
 * Returns a boolean which indicates whether it will be legal to assign
 * num to the given row,col location.
 */
int is_safe(int grid[N][N], int row, int col, int num)
{
    /* Check if 'num' is not already placed in current row,
       current column and current 3x3 box */
    return !used_in_row(grid, row, num) && 
           !used_in_col(grid, col, num) &&
           !used_in_box(grid, row - row % 3, col - col % 3, num) && 
           grid[row][col] == UNASSIGNED;
}

/* Returns a boolean which indicates whether any assigned entry
   in the specified row matches the given number. */
int used_in_row(int grid[N][N], int row, int num)
{
    for (int col = 0; col < N; col++)
        if (grid[row][col] == num)
            return 1;
    return 0;
}

/* Returns a boolean which indicates whether any assigned entry
   in the specified column matches the given number. */
int used_in_col(int grid[N][N], int col, int num)
{
    for (int row = 0; row < N; row++)
        if (grid[row][col] == num)
            return 1;
    return 0;
}

/* Returns a boolean which indicates whether any assigned entry
   within the specified 3x3 box matches the given number. */
int used_in_box(int grid[N][N], int box_start_row, int box_start_col, int num)
{
    for (int row = 0; row < 3; row++)
        for (int col = 0; col < 3; col++)
            if (grid[row + box_start_row][col + box_start_col] == num)
                return 1;
    return 0;
}

/* A utility function to print grid */
void print_grid(int grid[N][N])
{
    printf("\n+-------+-------+-------+\n");
    for (int row = 0; row < N; row++) {
        printf("| ");
        for (int col = 0; col < N; col++) {
            if (grid[row][col] == 0) {
                printf(". ");
            } else {
                printf("%d ", grid[row][col]);
            }
            if ((col + 1) % 3 == 0) {
                printf("| ");
            }
        }
        printf("\n");
        if ((row + 1) % 3 == 0) {
            printf("+-------+-------+-------+\n");
        }
    }
}

/* Load a sample puzzle for testing */
void load_example_puzzle(int grid[N][N])
{
    // Built-in example puzzle (easy difficulty)
    // Use 0 for empty cells, 1-9 for filled cells
    int example[N][N] = {
        {5, 3, 0, 0, 7, 0, 0, 0, 0},
        {6, 0, 0, 1, 9, 5, 0, 0, 0},
        {0, 9, 8, 0, 0, 0, 0, 6, 0},
        {8, 0, 0, 0, 6, 0, 0, 0, 3},
        {4, 0, 0, 8, 0, 3, 0, 0, 1},
        {7, 0, 0, 0, 2, 0, 0, 0, 6},
        {0, 6, 0, 0, 0, 0, 2, 8, 0},
        {0, 0, 0, 4, 1, 9, 0, 0, 5},
        {0, 0, 0, 0, 8, 0, 0, 7, 9}
    };
    
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            grid[i][j] = example[i][j];
        }
    }
}

/* ========== Helper Functions ========== */

/* Count the number of set bits (candidates) in a bitmask */
int count_candidates(candidates_t c)
{
    int count = 0;
    while (c) {
        count += c & 1;
        c >>= 1;
    }
    return count;
}

/* Return the first (lowest) candidate value, or 0 if none */
int first_candidate(candidates_t c)
{
    for (int n = 1; n <= 9; n++) {
        if (c & CANDIDATE(n))
            return n;
    }
    return 0;
}

/* Check if grid is completely solved (no unassigned cells) */
int is_solved(int grid[N][N])
{
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            if (grid[i][j] == UNASSIGNED)
                return 0;
    return 1;
}

/* ========== Constraint Propagation ========== */

/* Initialize candidates from current grid state */
void init_candidates(int grid[N][N], candidates_t cands[N][N])
{
    // Start with all candidates for empty cells, none for filled
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (grid[i][j] != UNASSIGNED) {
                cands[i][j] = 0;
            } else {
                cands[i][j] = ALL_CANDIDATES;
            }
        }
    }

    // Eliminate based on existing values
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (grid[i][j] != UNASSIGNED) {
                eliminate_from_peers(cands, i, j, grid[i][j]);
            }
        }
    }
}

/* Remove a candidate from all peers of a cell.
 * Returns 0 if any peer becomes empty (contradiction), 1 otherwise. */
int eliminate_from_peers(candidates_t cands[N][N], int row, int col, int num)
{
    candidates_t bit = CANDIDATE(num);

    // Eliminate from row
    for (int c = 0; c < N; c++) {
        if (c != col) {
            cands[row][c] &= ~bit;
            if (cands[row][c] == 0 && cands[row][c] != (candidates_t)-1)
                ; // Will check below
        }
    }

    // Eliminate from column
    for (int r = 0; r < N; r++) {
        if (r != row) {
            cands[r][col] &= ~bit;
        }
    }

    // Eliminate from 3x3 box
    int box_r = (row / 3) * 3;
    int box_c = (col / 3) * 3;
    for (int r = box_r; r < box_r + 3; r++) {
        for (int c = box_c; c < box_c + 3; c++) {
            if (r != row || c != col) {
                cands[r][c] &= ~bit;
            }
        }
    }

    return 1;
}

/* Propagate constraints: naked singles and hidden singles.
 * Returns number of cells filled, or -1 on contradiction. */
int propagate(int grid[N][N], candidates_t cands[N][N])
{
    int total_filled = 0;
    int progress = 1;

    while (progress) {
        progress = 0;

        // Naked singles: cells with exactly one candidate
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                if (grid[i][j] == UNASSIGNED) {
                    int count = count_candidates(cands[i][j]);
                    if (count == 0) {
                        return -1; // Contradiction
                    }
                    if (count == 1) {
                        int val = first_candidate(cands[i][j]);
                        grid[i][j] = val;
                        cands[i][j] = 0;
                        eliminate_from_peers(cands, i, j, val);
                        total_filled++;
                        progress = 1;
                    }
                }
            }
        }

        // Hidden singles in rows
        for (int row = 0; row < N; row++) {
            for (int num = 1; num <= 9; num++) {
                int count = 0;
                int last_col = -1;
                for (int col = 0; col < N; col++) {
                    if (grid[row][col] == num) {
                        count = -1; // Already placed
                        break;
                    }
                    if (grid[row][col] == UNASSIGNED && (cands[row][col] & CANDIDATE(num))) {
                        count++;
                        last_col = col;
                    }
                }
                if (count == 0) return -1; // No place for this number
                if (count == 1 && last_col >= 0) {
                    grid[row][last_col] = num;
                    cands[row][last_col] = 0;
                    eliminate_from_peers(cands, row, last_col, num);
                    total_filled++;
                    progress = 1;
                }
            }
        }

        // Hidden singles in columns
        for (int col = 0; col < N; col++) {
            for (int num = 1; num <= 9; num++) {
                int count = 0;
                int last_row = -1;
                for (int row = 0; row < N; row++) {
                    if (grid[row][col] == num) {
                        count = -1;
                        break;
                    }
                    if (grid[row][col] == UNASSIGNED && (cands[row][col] & CANDIDATE(num))) {
                        count++;
                        last_row = row;
                    }
                }
                if (count == 0) return -1;
                if (count == 1 && last_row >= 0) {
                    grid[last_row][col] = num;
                    cands[last_row][col] = 0;
                    eliminate_from_peers(cands, last_row, col, num);
                    total_filled++;
                    progress = 1;
                }
            }
        }

        // Hidden singles in 3x3 boxes
        for (int box_r = 0; box_r < 9; box_r += 3) {
            for (int box_c = 0; box_c < 9; box_c += 3) {
                for (int num = 1; num <= 9; num++) {
                    int count = 0;
                    int last_r = -1, last_c = -1;
                    for (int r = box_r; r < box_r + 3; r++) {
                        for (int c = box_c; c < box_c + 3; c++) {
                            if (grid[r][c] == num) {
                                count = -1;
                                break;
                            }
                            if (grid[r][c] == UNASSIGNED && (cands[r][c] & CANDIDATE(num))) {
                                count++;
                                last_r = r;
                                last_c = c;
                            }
                        }
                        if (count == -1) break;
                    }
                    if (count == 0) return -1;
                    if (count == 1 && last_r >= 0) {
                        grid[last_r][last_c] = num;
                        cands[last_r][last_c] = 0;
                        eliminate_from_peers(cands, last_r, last_c, num);
                        total_filled++;
                        progress = 1;
                    }
                }
            }
        }
    }

    return total_filled;
}

/*
 * Validate a sudoku grid for correctness.
 * Checks:
 *   - All values are in range 0-9
 *   - No duplicate values in any row
 *   - No duplicate values in any column
 *   - No duplicate values in any 3x3 box
 *
 * Returns 1 if valid, 0 if invalid.
 */
int validate_grid(int grid[N][N])
{
    // Check each cell for valid range
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (grid[i][j] < 0 || grid[i][j] > 9)
                return 0;
        }
    }

    // Check rows for duplicates
    for (int row = 0; row < N; row++) {
        int seen = 0;
        for (int col = 0; col < N; col++) {
            int val = grid[row][col];
            if (val != UNASSIGNED) {
                if (seen & CANDIDATE(val))
                    return 0; // Duplicate
                seen |= CANDIDATE(val);
            }
        }
    }

    // Check columns for duplicates
    for (int col = 0; col < N; col++) {
        int seen = 0;
        for (int row = 0; row < N; row++) {
            int val = grid[row][col];
            if (val != UNASSIGNED) {
                if (seen & CANDIDATE(val))
                    return 0; // Duplicate
                seen |= CANDIDATE(val);
            }
        }
    }

    // Check 3x3 boxes for duplicates
    for (int box_r = 0; box_r < 9; box_r += 3) {
        for (int box_c = 0; box_c < 9; box_c += 3) {
            int seen = 0;
            for (int r = box_r; r < box_r + 3; r++) {
                for (int c = box_c; c < box_c + 3; c++) {
                    int val = grid[r][c];
                    if (val != UNASSIGNED) {
                        if (seen & CANDIDATE(val))
                            return 0; // Duplicate
                        seen |= CANDIDATE(val);
                    }
                }
            }
        }
    }

    return 1; // Valid
}

/*
 * Load a sudoku puzzle from a file.
 * Format: 9 lines of 9 digits (0-9), where 0 means empty.
 * Whitespace and non-digit characters are ignored.
 *
 * Returns 1 on success, 0 on failure.
 */
int load_from_file(const char* filename, int grid[N][N])
{
    FILE* f = fopen(filename, "r");
    if (!f) return 0;

    int count = 0;
    int c;
    while ((c = fgetc(f)) != EOF && count < 81) {
        if (c >= '0' && c <= '9') {
            grid[count / N][count % N] = c - '0';
            count++;
        }
        // Skip non-digit characters (whitespace, newlines, etc.)
    }
    fclose(f);

    return (count == 81) ? 1 : 0;
}

/*
 * Save a sudoku puzzle to a file.
 * Format: 9 lines of 9 digits, one row per line.
 *
 * Returns 1 on success, 0 on failure.
 */
int save_to_file(const char* filename, int grid[N][N])
{
    FILE* f = fopen(filename, "w");
    if (!f) return 0;

    for (int row = 0; row < N; row++) {
        for (int col = 0; col < N; col++) {
            fputc('0' + grid[row][col], f);
        }
        fputc('\n', f);
    }
    fclose(f);

    return 1;
}
