/*
 * Unit tests for the Sudoku solver
 */

#include <stdio.h>
#include <string.h>
#include <time.h>
#include "minunit.h"
#include "../include/sudoku.h"

/* ========== Test Data ========== */

/* Easy puzzle - solvable by naked singles only */
static int puzzle_easy[N][N] = {
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

/* Easy puzzle solution */
static int solution_easy[N][N] = {
    {5, 3, 4, 6, 7, 8, 9, 1, 2},
    {6, 7, 2, 1, 9, 5, 3, 4, 8},
    {1, 9, 8, 3, 4, 2, 5, 6, 7},
    {8, 5, 9, 7, 6, 1, 4, 2, 3},
    {4, 2, 6, 8, 5, 3, 7, 9, 1},
    {7, 1, 3, 9, 2, 4, 8, 5, 6},
    {9, 6, 1, 5, 3, 7, 2, 8, 4},
    {2, 8, 7, 4, 1, 9, 6, 3, 5},
    {3, 4, 5, 2, 8, 6, 1, 7, 9}
};

/* Hard puzzle - requires backtracking */
static int puzzle_hard[N][N] = {
    {0, 0, 0, 6, 0, 0, 4, 0, 0},
    {7, 0, 0, 0, 0, 3, 6, 0, 0},
    {0, 0, 0, 0, 9, 1, 0, 8, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 5, 0, 1, 8, 0, 0, 0, 3},
    {0, 0, 0, 3, 0, 6, 0, 4, 5},
    {0, 4, 0, 2, 0, 0, 0, 6, 0},
    {9, 0, 3, 0, 0, 0, 0, 0, 0},
    {0, 2, 0, 0, 0, 0, 1, 0, 0}
};

/* Evil/extreme puzzle - many backtracks needed */
static int puzzle_evil[N][N] = {
    {0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 3, 0, 8, 5},
    {0, 0, 1, 0, 2, 0, 0, 0, 0},
    {0, 0, 0, 5, 0, 7, 0, 0, 0},
    {0, 0, 4, 0, 0, 0, 1, 0, 0},
    {0, 9, 0, 0, 0, 0, 0, 0, 0},
    {5, 0, 0, 0, 0, 0, 0, 7, 3},
    {0, 0, 2, 0, 1, 0, 0, 0, 0},
    {0, 0, 0, 0, 4, 0, 0, 0, 9}
};

/* Minimal puzzle - 17 clues (minimum for unique solution) */
static int puzzle_minimal[N][N] = {
    {0, 0, 0, 0, 0, 0, 0, 1, 0},
    {4, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 2, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 5, 0, 4, 0, 7},
    {0, 0, 8, 0, 0, 0, 3, 0, 0},
    {0, 0, 1, 0, 9, 0, 0, 0, 0},
    {3, 0, 0, 4, 0, 0, 2, 0, 0},
    {0, 5, 0, 1, 0, 0, 0, 0, 0},
    {0, 0, 0, 8, 0, 6, 0, 0, 0}
};

/* Almost complete - only one empty cell */
static int puzzle_almost_complete[N][N] = {
    {5, 3, 4, 6, 7, 8, 9, 1, 2},
    {6, 7, 2, 1, 9, 5, 3, 4, 8},
    {1, 9, 8, 3, 4, 2, 5, 6, 7},
    {8, 5, 9, 7, 6, 1, 4, 2, 3},
    {4, 2, 6, 8, 5, 3, 7, 9, 1},
    {7, 1, 3, 9, 2, 4, 8, 5, 6},
    {9, 6, 1, 5, 3, 7, 2, 8, 4},
    {2, 8, 7, 4, 1, 9, 6, 3, 5},
    {3, 4, 5, 2, 8, 6, 1, 7, 0}  /* Last cell empty */
};

/* Invalid: duplicate in row */
static int puzzle_invalid_row[N][N] = {
    {5, 3, 5, 0, 7, 0, 0, 0, 0},  /* 5 appears twice in row 0 */
    {6, 0, 0, 1, 9, 5, 0, 0, 0},
    {0, 9, 8, 0, 0, 0, 0, 6, 0},
    {8, 0, 0, 0, 6, 0, 0, 0, 3},
    {4, 0, 0, 8, 0, 3, 0, 0, 1},
    {7, 0, 0, 0, 2, 0, 0, 0, 6},
    {0, 6, 0, 0, 0, 0, 2, 8, 0},
    {0, 0, 0, 4, 1, 9, 0, 0, 5},
    {0, 0, 0, 0, 8, 0, 0, 7, 9}
};

/* Invalid: duplicate in column */
static int puzzle_invalid_col[N][N] = {
    {5, 3, 0, 0, 7, 0, 0, 0, 0},
    {6, 0, 0, 1, 9, 5, 0, 0, 0},
    {0, 9, 8, 0, 0, 0, 0, 6, 0},
    {8, 0, 0, 0, 6, 0, 0, 0, 3},
    {4, 0, 0, 8, 0, 3, 0, 0, 1},
    {7, 0, 0, 0, 2, 0, 0, 0, 6},
    {0, 6, 0, 0, 0, 0, 2, 8, 0},
    {5, 0, 0, 4, 1, 9, 0, 0, 5},  /* 5 appears twice in col 0 */
    {0, 0, 0, 0, 8, 0, 0, 7, 9}
};

/* Invalid: duplicate in 3x3 box */
static int puzzle_invalid_box[N][N] = {
    {5, 3, 0, 0, 7, 0, 0, 0, 0},
    {6, 0, 5, 1, 9, 5, 0, 0, 0},  /* 5 in box 0 twice (row0col0, row1col2) */
    {0, 9, 8, 0, 0, 0, 0, 6, 0},
    {8, 0, 0, 0, 6, 0, 0, 0, 3},
    {4, 0, 0, 8, 0, 3, 0, 0, 1},
    {7, 0, 0, 0, 2, 0, 0, 0, 6},
    {0, 6, 0, 0, 0, 0, 2, 8, 0},
    {0, 0, 0, 4, 1, 9, 0, 0, 5},
    {0, 0, 0, 0, 8, 0, 0, 7, 9}
};

/* Invalid: out of range value */
static int puzzle_out_of_range[N][N] = {
    {5, 3, 0, 0, 7, 0, 0, 0, 0},
    {6, 0, 0, 1, 9, 5, 0, 0, 0},
    {0, 9, 8, 0, 0, 0, 0, 6, 0},
    {8, 0, 0, 0, 6, 0, 0, 0, 3},
    {4, 0, 0, 8, 10, 3, 0, 0, 1},  /* 10 is out of range */
    {7, 0, 0, 0, 2, 0, 0, 0, 6},
    {0, 6, 0, 0, 0, 0, 2, 8, 0},
    {0, 0, 0, 4, 1, 9, 0, 0, 5},
    {0, 0, 0, 0, 8, 0, 0, 7, 9}
};

/* Unsolvable puzzle - valid input but no solution */
static int puzzle_unsolvable[N][N] = {
    {5, 1, 6, 8, 4, 9, 7, 3, 2},
    {3, 0, 7, 6, 0, 5, 0, 0, 0},
    {8, 0, 9, 7, 0, 0, 0, 6, 5},
    {1, 3, 5, 0, 6, 0, 9, 0, 7},
    {4, 7, 2, 5, 9, 1, 0, 0, 6},
    {9, 6, 8, 3, 7, 0, 0, 5, 0},
    {2, 5, 3, 1, 8, 6, 0, 7, 4},
    {6, 8, 4, 2, 0, 7, 5, 0, 0},
    {7, 9, 1, 0, 5, 0, 6, 0, 8}
};

/* Empty grid */
static int puzzle_empty[N][N] = {{0}};

/* Already solved grid */
static int puzzle_solved[N][N] = {
    {5, 3, 4, 6, 7, 8, 9, 1, 2},
    {6, 7, 2, 1, 9, 5, 3, 4, 8},
    {1, 9, 8, 3, 4, 2, 5, 6, 7},
    {8, 5, 9, 7, 6, 1, 4, 2, 3},
    {4, 2, 6, 8, 5, 3, 7, 9, 1},
    {7, 1, 3, 9, 2, 4, 8, 5, 6},
    {9, 6, 1, 5, 3, 7, 2, 8, 4},
    {2, 8, 7, 4, 1, 9, 6, 3, 5},
    {3, 4, 5, 2, 8, 6, 1, 7, 9}
};

/* ========== Helper Functions ========== */

static void copy_grid(int dest[N][N], int src[N][N]) {
    memcpy(dest, src, sizeof(int) * N * N);
}

static int grids_equal(int a[N][N], int b[N][N]) {
    return memcmp(a, b, sizeof(int) * N * N) == 0;
}

/* Verify a solved grid is valid (all constraints satisfied) */
static int verify_solution(int grid[N][N]) {
    /* Check all cells filled */
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (grid[i][j] < 1 || grid[i][j] > 9)
                return 0;
        }
    }

    /* Check rows */
    for (int row = 0; row < N; row++) {
        int seen = 0;
        for (int col = 0; col < N; col++) {
            int bit = 1 << grid[row][col];
            if (seen & bit) return 0;
            seen |= bit;
        }
    }

    /* Check columns */
    for (int col = 0; col < N; col++) {
        int seen = 0;
        for (int row = 0; row < N; row++) {
            int bit = 1 << grid[row][col];
            if (seen & bit) return 0;
            seen |= bit;
        }
    }

    /* Check 3x3 boxes */
    for (int box_r = 0; box_r < 9; box_r += 3) {
        for (int box_c = 0; box_c < 9; box_c += 3) {
            int seen = 0;
            for (int r = box_r; r < box_r + 3; r++) {
                for (int c = box_c; c < box_c + 3; c++) {
                    int bit = 1 << grid[r][c];
                    if (seen & bit) return 0;
                    seen |= bit;
                }
            }
        }
    }

    return 1;
}

/* ========== Validation Tests ========== */

static char* test_validate_valid_grid(void) {
    int grid[N][N];
    copy_grid(grid, puzzle_easy);
    mu_assert("valid grid should pass validation", validate_grid(grid) == 1);
    return NULL;
}

static char* test_validate_row_duplicate(void) {
    int grid[N][N];
    copy_grid(grid, puzzle_invalid_row);
    mu_assert("row duplicate should fail validation", validate_grid(grid) == 0);
    return NULL;
}

static char* test_validate_col_duplicate(void) {
    int grid[N][N];
    copy_grid(grid, puzzle_invalid_col);
    mu_assert("column duplicate should fail validation", validate_grid(grid) == 0);
    return NULL;
}

static char* test_validate_box_duplicate(void) {
    int grid[N][N];
    copy_grid(grid, puzzle_invalid_box);
    mu_assert("box duplicate should fail validation", validate_grid(grid) == 0);
    return NULL;
}

static char* test_validate_out_of_range(void) {
    int grid[N][N];
    copy_grid(grid, puzzle_out_of_range);
    mu_assert("out of range value should fail validation", validate_grid(grid) == 0);
    return NULL;
}

static char* test_validate_negative_value(void) {
    int grid[N][N];
    copy_grid(grid, puzzle_easy);
    grid[0][0] = -1;
    mu_assert("negative value should fail validation", validate_grid(grid) == 0);
    return NULL;
}

static char* test_validate_empty_grid(void) {
    int grid[N][N];
    copy_grid(grid, puzzle_empty);
    mu_assert("empty grid should be valid", validate_grid(grid) == 1);
    return NULL;
}

static char* test_validate_solved_grid(void) {
    int grid[N][N];
    copy_grid(grid, puzzle_solved);
    mu_assert("solved grid should be valid", validate_grid(grid) == 1);
    return NULL;
}

/* ========== Solving Tests ========== */

static char* test_solve_easy_puzzle(void) {
    int grid[N][N];
    copy_grid(grid, puzzle_easy);
    int result = solve_sudoku(grid);
    mu_assert("easy puzzle should return SOLVE_SUCCESS", result == SOLVE_SUCCESS);
    mu_assert("easy puzzle solution should match expected", grids_equal(grid, solution_easy));
    mu_assert("easy puzzle solution should verify", verify_solution(grid));
    return NULL;
}

static char* test_solve_hard_puzzle(void) {
    int grid[N][N];
    copy_grid(grid, puzzle_hard);
    int result = solve_sudoku(grid);
    mu_assert("hard puzzle should return SOLVE_SUCCESS", result == SOLVE_SUCCESS);
    mu_assert("hard puzzle solution should verify", verify_solution(grid));
    return NULL;
}

static char* test_solve_evil_puzzle(void) {
    int grid[N][N];
    copy_grid(grid, puzzle_evil);
    int result = solve_sudoku(grid);
    mu_assert("evil puzzle should return SOLVE_SUCCESS", result == SOLVE_SUCCESS);
    mu_assert("evil puzzle solution should verify", verify_solution(grid));
    return NULL;
}

static char* test_solve_minimal_puzzle(void) {
    int grid[N][N];
    copy_grid(grid, puzzle_minimal);
    int result = solve_sudoku(grid);
    mu_assert("minimal puzzle should return SOLVE_SUCCESS", result == SOLVE_SUCCESS);
    mu_assert("minimal puzzle solution should verify", verify_solution(grid));
    return NULL;
}

static char* test_solve_almost_complete(void) {
    int grid[N][N];
    copy_grid(grid, puzzle_almost_complete);
    int result = solve_sudoku(grid);
    mu_assert("almost complete puzzle should return SOLVE_SUCCESS", result == SOLVE_SUCCESS);
    mu_assert_int_eq(9, grid[8][8]);  /* The missing cell should be 9 */
    mu_assert("almost complete solution should verify", verify_solution(grid));
    return NULL;
}

static char* test_solve_empty_grid(void) {
    int grid[N][N];
    copy_grid(grid, puzzle_empty);
    int result = solve_sudoku(grid);
    mu_assert("empty grid should return SOLVE_SUCCESS", result == SOLVE_SUCCESS);
    mu_assert("empty grid solution should verify", verify_solution(grid));
    return NULL;
}

static char* test_solve_already_solved(void) {
    int grid[N][N];
    int original[N][N];
    copy_grid(grid, puzzle_solved);
    copy_grid(original, puzzle_solved);
    int result = solve_sudoku(grid);
    mu_assert("already solved should return SOLVE_SUCCESS", result == SOLVE_SUCCESS);
    mu_assert("already solved should remain unchanged", grids_equal(grid, original));
    return NULL;
}

static char* test_solve_unsolvable(void) {
    int grid[N][N];
    copy_grid(grid, puzzle_unsolvable);
    int result = solve_sudoku(grid);
    mu_assert("unsolvable puzzle should return SOLVE_NOSOLUTION", result == SOLVE_NOSOLUTION);
    return NULL;
}

static char* test_solve_invalid_row_dup(void) {
    int grid[N][N];
    copy_grid(grid, puzzle_invalid_row);
    int result = solve_sudoku(grid);
    mu_assert("row duplicate should return SOLVE_INVALID", result == SOLVE_INVALID);
    return NULL;
}

static char* test_solve_invalid_col_dup(void) {
    int grid[N][N];
    copy_grid(grid, puzzle_invalid_col);
    int result = solve_sudoku(grid);
    mu_assert("column duplicate should return SOLVE_INVALID", result == SOLVE_INVALID);
    return NULL;
}

static char* test_solve_invalid_box_dup(void) {
    int grid[N][N];
    copy_grid(grid, puzzle_invalid_box);
    int result = solve_sudoku(grid);
    mu_assert("box duplicate should return SOLVE_INVALID", result == SOLVE_INVALID);
    return NULL;
}

static char* test_solve_invalid_out_of_range(void) {
    int grid[N][N];
    copy_grid(grid, puzzle_out_of_range);
    int result = solve_sudoku(grid);
    mu_assert("out of range should return SOLVE_INVALID", result == SOLVE_INVALID);
    return NULL;
}

/* ========== Helper Function Tests ========== */

static char* test_find_unassigned_location(void) {
    int grid[N][N];
    copy_grid(grid, puzzle_easy);
    int row, col;
    int found = find_unassigned_location(grid, &row, &col);
    mu_assert("should find unassigned location", found == 1);
    mu_assert("found location should be unassigned", grid[row][col] == UNASSIGNED);
    return NULL;
}

static char* test_find_unassigned_none(void) {
    int grid[N][N];
    copy_grid(grid, puzzle_solved);
    int row, col;
    int found = find_unassigned_location(grid, &row, &col);
    mu_assert("should not find unassigned in solved grid", found == 0);
    return NULL;
}

static char* test_is_safe(void) {
    int grid[N][N];
    copy_grid(grid, puzzle_easy);
    /* Cell [0][2] is empty, and based on constraints, 4 should be valid there */
    mu_assert("4 should be safe at [0][2]", is_safe(grid, 0, 2, 4) == 1);
    /* 5 is already in row 0 */
    mu_assert("5 should not be safe at [0][2] (row conflict)", is_safe(grid, 0, 2, 5) == 0);
    /* 8 is already in column 2 */
    mu_assert("8 should not be safe at [0][2] (col conflict)", is_safe(grid, 0, 2, 8) == 0);
    return NULL;
}

static char* test_is_solved(void) {
    int grid[N][N];
    copy_grid(grid, puzzle_solved);
    mu_assert("solved grid should return is_solved=1", is_solved(grid) == 1);

    copy_grid(grid, puzzle_easy);
    mu_assert("unsolved grid should return is_solved=0", is_solved(grid) == 0);
    return NULL;
}

static char* test_count_candidates(void) {
    mu_assert_int_eq(0, count_candidates(0));
    mu_assert_int_eq(1, count_candidates(CANDIDATE(5)));
    mu_assert_int_eq(2, count_candidates(CANDIDATE(3) | CANDIDATE(7)));
    mu_assert_int_eq(9, count_candidates(ALL_CANDIDATES));
    return NULL;
}

static char* test_first_candidate(void) {
    mu_assert_int_eq(0, first_candidate(0));
    mu_assert_int_eq(5, first_candidate(CANDIDATE(5)));
    mu_assert_int_eq(3, first_candidate(CANDIDATE(3) | CANDIDATE(7)));
    mu_assert_int_eq(1, first_candidate(ALL_CANDIDATES));
    return NULL;
}

/* ========== Performance Tests ========== */

static char* test_performance_hard(void) {
    int grid[N][N];
    copy_grid(grid, puzzle_hard);

    clock_t start = clock();
    int result = solve_sudoku(grid);
    clock_t end = clock();

    double time_ms = (double)(end - start) * 1000.0 / CLOCKS_PER_SEC;

    mu_assert("hard puzzle should solve", result == SOLVE_SUCCESS);

    /* Should solve in under 100ms */
    if (time_ms > 100.0) {
        static char msg[128];
        snprintf(msg, sizeof(msg), "hard puzzle took %.2fms (limit: 100ms)", time_ms);
        return msg;
    }

    printf(" (%.2fms)", time_ms);
    return NULL;
}

static char* test_performance_evil(void) {
    int grid[N][N];
    copy_grid(grid, puzzle_evil);

    clock_t start = clock();
    int result = solve_sudoku(grid);
    clock_t end = clock();

    double time_ms = (double)(end - start) * 1000.0 / CLOCKS_PER_SEC;

    mu_assert("evil puzzle should solve", result == SOLVE_SUCCESS);

    /* Should solve in under 1 second */
    if (time_ms > 1000.0) {
        static char msg[128];
        snprintf(msg, sizeof(msg), "evil puzzle took %.2fms (limit: 1000ms)", time_ms);
        return msg;
    }

    printf(" (%.2fms)", time_ms);
    return NULL;
}

/* ========== File I/O Tests ========== */

static char* test_load_from_file(void) {
    int grid[N][N];
    int result = load_from_file("tests/data/valid_easy.sudoku", grid);
    mu_assert("should load valid file", result == 1);
    mu_assert("loaded grid should be valid", validate_grid(grid) == 1);
    return NULL;
}

static char* test_load_nonexistent_file(void) {
    int grid[N][N];
    int result = load_from_file("nonexistent_file.sudoku", grid);
    mu_assert("should fail on nonexistent file", result == 0);
    return NULL;
}

static char* test_save_and_load(void) {
    int original[N][N];
    int loaded[N][N];
    copy_grid(original, puzzle_easy);

    int save_result = save_to_file("/tmp/test_puzzle.sudoku", original);
    mu_assert("should save successfully", save_result == 1);

    int load_result = load_from_file("/tmp/test_puzzle.sudoku", loaded);
    mu_assert("should load successfully", load_result == 1);

    mu_assert("loaded should match original", grids_equal(original, loaded));
    return NULL;
}

/* ========== Main Test Runner ========== */

static void run_all_tests(void) {
    printf("\n=== Validation Tests ===\n");
    mu_run_test(test_validate_valid_grid);
    mu_run_test(test_validate_row_duplicate);
    mu_run_test(test_validate_col_duplicate);
    mu_run_test(test_validate_box_duplicate);
    mu_run_test(test_validate_out_of_range);
    mu_run_test(test_validate_negative_value);
    mu_run_test(test_validate_empty_grid);
    mu_run_test(test_validate_solved_grid);

    printf("\n=== Solving Tests ===\n");
    mu_run_test(test_solve_easy_puzzle);
    mu_run_test(test_solve_hard_puzzle);
    mu_run_test(test_solve_evil_puzzle);
    mu_run_test(test_solve_minimal_puzzle);
    mu_run_test(test_solve_almost_complete);
    mu_run_test(test_solve_empty_grid);
    mu_run_test(test_solve_already_solved);
    mu_run_test(test_solve_unsolvable);
    mu_run_test(test_solve_invalid_row_dup);
    mu_run_test(test_solve_invalid_col_dup);
    mu_run_test(test_solve_invalid_box_dup);
    mu_run_test(test_solve_invalid_out_of_range);

    printf("\n=== Helper Function Tests ===\n");
    mu_run_test(test_find_unassigned_location);
    mu_run_test(test_find_unassigned_none);
    mu_run_test(test_is_safe);
    mu_run_test(test_is_solved);
    mu_run_test(test_count_candidates);
    mu_run_test(test_first_candidate);

    printf("\n=== Performance Tests ===\n");
    mu_run_test(test_performance_hard);
    mu_run_test(test_performance_evil);

    printf("\n=== File I/O Tests ===\n");
    mu_run_test(test_load_from_file);
    mu_run_test(test_load_nonexistent_file);
    mu_run_test(test_save_and_load);
}

int main(void) {
    printf("Sudoku Solver Unit Tests\n");
    printf("========================\n");

    run_all_tests();

    mu_print_summary();

    return mu_exit_code();
}
