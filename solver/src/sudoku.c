#include <stdio.h>
#include "../include/sudoku.h"

/* 
 * Takes a partially filled-in grid and attempts to assign values to
 * all unassigned locations in such a way to meet the requirements
 * for Sudoku (non-duplication across rows, columns, and boxes)
 */
int solve_sudoku(int grid[N][N])
{
    int row, col;

    /* If there is no unassigned location, we are done */
    if (!find_unassigned_location(grid, &row, &col))
        return 1; // success!

    /* Consider digits 1 to 9 */
    for (int num = 1; num <= 9; num++) {
        /* If looks promising */
        if (is_safe(grid, row, col, num)) {
            /* Make tentative assignment */
            grid[row][col] = num;

            /* Return, if success, yay! */
            if (solve_sudoku(grid))
                return 1;

            /* Failure, unmake & try again */
            grid[row][col] = UNASSIGNED;
        }
    }
    return 0; // This triggers backtracking
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