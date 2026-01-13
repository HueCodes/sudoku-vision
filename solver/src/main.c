#include <stdio.h>
#include <stdlib.h>
#include "../include/sudoku.h"

void input_puzzle(int grid[N][N])
{
    printf("\nEnter your Sudoku puzzle:\n");
    printf("Use 0 for empty cells, 1-9 for filled cells\n");
    printf("Enter 81 numbers (9 rows × 9 columns):\n\n");
    
    for (int i = 0; i < N; i++) {
        printf("Row %d: ", i + 1);
        for (int j = 0; j < N; j++) {
            scanf("%d", &grid[i][j]);
            // Validate input
            if (grid[i][j] < 0 || grid[i][j] > 9) {
                printf("Invalid input! Please enter numbers 0-9 only.\n");
                j--; // Retry this cell
            }
        }
    }
}

void input_puzzle_line_by_line(int grid[N][N])
{
    printf("\nEnter your Sudoku puzzle row by row:\n");
    printf("Use 0 for empty cells, 1-9 for filled cells\n");
    printf("Enter 9 numbers per row, separated by spaces:\n\n");
    
    for (int i = 0; i < N; i++) {
        printf("Row %d (9 numbers): ", i + 1);
        for (int j = 0; j < N; j++) {
            if (scanf("%d", &grid[i][j]) != 1 || grid[i][j] < 0 || grid[i][j] > 9) {
                printf("Invalid input! Please enter numbers 0-9 only.\n");
                printf("Row %d (9 numbers): ", i + 1);
                j = -1; // Restart this row
                // Clear input buffer
                int c;
                while ((c = getchar()) != '\n' && c != EOF);
            }
        }
    }
}

int main(void)
{
    int grid[N][N];
    int choice;
    
    printf("=== C Sudoku Solver ===\n");
    printf("Using Backtracking Algorithm\n\n");
    
    printf("Choose an option:\n");
    printf("1. Use built-in example puzzle\n");
    printf("2. Enter your own puzzle (all at once)\n");
    printf("3. Enter your own puzzle (row by row)\n");
    printf("Enter choice (1-3): ");
    
    scanf("%d", &choice);
    
    switch (choice) {
        case 1:
            load_example_puzzle(grid);
            printf("\nUsing built-in example puzzle:");
            break;
        case 2:
            input_puzzle(grid);
            printf("\nYour puzzle:");
            break;
        case 3:
            input_puzzle_line_by_line(grid);
            printf("\nYour puzzle:");
            break;
        default:
            printf("Invalid choice! Using built-in example puzzle.\n");
            load_example_puzzle(grid);
            printf("\nUsing built-in example puzzle:");
            break;
    }
    
    print_grid(grid);
    
    printf("\nSolving...\n");
    
    if (solve_sudoku(grid) == 1) {
        printf("\nSolution found!");
        print_grid(grid);
    } else {
        printf("\nNo solution exists for this puzzle.\n");
        printf("Please check that you entered the puzzle correctly.\n");
    }
    
    return 0;
}