#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "../include/sudoku.h"

static void print_usage(const char* progname)
{
    printf("Usage: %s [options] [input_file]\n", progname);
    printf("\nOptions:\n");
    printf("  -o FILE    Write solution to FILE\n");
    printf("  -b         Run benchmark suite\n");
    printf("  -h         Show this help\n");
    printf("\nIf no input file is given, runs in interactive mode.\n");
    printf("File format: 81 digits (0-9), whitespace ignored. 0 = empty cell.\n");
}

// Benchmark puzzles of varying difficulty
static const char* benchmark_puzzles[] = {
    // Easy: mostly naked singles
    "530070000600195000098000060800060003400803001700020006060000280000419005000080079",
    // Medium: requires some hidden singles
    "000000010400000000020000000000050407008000300001090000300400200050100000000806000",
    // Hard: Arto Inkala's "World's Hardest Sudoku"
    "800000000003600000070090200050007000000045700000100030001000068008500010090000400",
    // Evil: requires significant backtracking
    "000000000000003085001020000000507000004000100090000000500000073002010000000040009",
    NULL
};

static const char* benchmark_names[] = {
    "Easy", "Medium", "Hard", "Evil"
};

static void load_puzzle_from_string(const char* str, int grid[N][N])
{
    for (int i = 0; i < 81; i++) {
        grid[i / 9][i % 9] = str[i] - '0';
    }
}

static int run_benchmark(void)
{
    printf("=== Sudoku Solver Benchmark ===\n\n");

    int total_puzzles = 0;
    double total_time = 0;

    for (int p = 0; benchmark_puzzles[p] != NULL; p++) {
        int grid[N][N];
        load_puzzle_from_string(benchmark_puzzles[p], grid);

        printf("%-8s: ", benchmark_names[p]);
        fflush(stdout);

        // Run multiple iterations for more accurate timing
        int iterations = 100;
        clock_t start = clock();

        for (int i = 0; i < iterations; i++) {
            load_puzzle_from_string(benchmark_puzzles[p], grid);
            solve_sudoku(grid);
        }

        clock_t end = clock();
        double elapsed_ms = ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0;
        double per_solve_us = (elapsed_ms * 1000.0) / iterations;

        printf("%8.1f us/solve  (%d iterations, %.1f ms total)\n",
               per_solve_us, iterations, elapsed_ms);

        total_puzzles += iterations;
        total_time += elapsed_ms;
    }

    printf("\n");
    printf("Total: %d solves in %.1f ms (%.1f solves/sec)\n",
           total_puzzles, total_time, (total_puzzles / total_time) * 1000.0);

    return 0;
}

static void input_puzzle_interactive(int grid[N][N])
{
    printf("\nEnter your Sudoku puzzle row by row:\n");
    printf("Use 0 for empty cells, 1-9 for filled cells\n");
    printf("Enter 9 numbers per row, separated by spaces:\n\n");

    for (int i = 0; i < N; i++) {
        printf("Row %d: ", i + 1);
        for (int j = 0; j < N; j++) {
            if (scanf("%d", &grid[i][j]) != 1 || grid[i][j] < 0 || grid[i][j] > 9) {
                printf("Invalid input! Please enter numbers 0-9 only.\n");
                printf("Row %d: ", i + 1);
                j = -1;
                int c;
                while ((c = getchar()) != '\n' && c != EOF);
            }
        }
    }
}

static int run_interactive(void)
{
    int grid[N][N];
    int choice;

    printf("=== C Sudoku Solver ===\n\n");
    printf("Choose an option:\n");
    printf("1. Use built-in example puzzle\n");
    printf("2. Enter your own puzzle\n");
    printf("Enter choice (1-2): ");

    scanf("%d", &choice);

    switch (choice) {
        case 1:
            load_example_puzzle(grid);
            printf("\nUsing built-in example puzzle:");
            break;
        case 2:
            input_puzzle_interactive(grid);
            printf("\nYour puzzle:");
            break;
        default:
            printf("Invalid choice! Using built-in example puzzle.\n");
            load_example_puzzle(grid);
            break;
    }

    print_grid(grid);
    printf("\nSolving...\n");

    int result = solve_sudoku(grid);
    switch (result) {
        case SOLVE_SUCCESS:
            printf("\nSolution found!");
            print_grid(grid);
            break;
        case SOLVE_NOSOLUTION:
            printf("\nNo solution exists for this puzzle.\n");
            break;
        case SOLVE_INVALID:
            printf("\nInvalid puzzle: contains duplicates or out-of-range values.\n");
            break;
    }

    return (result == SOLVE_SUCCESS) ? 0 : 1;
}

int main(int argc, char* argv[])
{
    const char* input_file = NULL;
    const char* output_file = NULL;
    int benchmark_mode = 0;

    // Parse arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        } else if (strcmp(argv[i], "-b") == 0 || strcmp(argv[i], "--benchmark") == 0) {
            benchmark_mode = 1;
        } else if (strcmp(argv[i], "-o") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: -o requires a filename\n");
                return 1;
            }
            output_file = argv[++i];
        } else if (argv[i][0] == '-') {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        } else {
            input_file = argv[i];
        }
    }

    if (benchmark_mode) {
        return run_benchmark();
    }

    // Interactive mode if no input file
    if (!input_file) {
        return run_interactive();
    }

    // File mode
    int grid[N][N];

    if (!load_from_file(input_file, grid)) {
        fprintf(stderr, "Error: Could not load puzzle from '%s'\n", input_file);
        return 1;
    }

    printf("Loaded puzzle from %s:\n", input_file);
    print_grid(grid);
    printf("\nSolving...\n");

    int result = solve_sudoku(grid);
    switch (result) {
        case SOLVE_SUCCESS:
            printf("\nSolution found!");
            print_grid(grid);
            if (output_file) {
                if (save_to_file(output_file, grid)) {
                    printf("\nSolution saved to %s\n", output_file);
                } else {
                    fprintf(stderr, "Error: Could not write to '%s'\n", output_file);
                    return 1;
                }
            }
            break;
        case SOLVE_NOSOLUTION:
            printf("\nNo solution exists for this puzzle.\n");
            break;
        case SOLVE_INVALID:
            printf("\nInvalid puzzle: contains duplicates or out-of-range values.\n");
            break;
    }

    return (result == SOLVE_SUCCESS) ? 0 : 1;
}
