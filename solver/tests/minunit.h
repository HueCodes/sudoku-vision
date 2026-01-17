/*
 * MinUnit - Minimal Unit Testing Framework for C
 * Based on: http://www.jera.com/techinfo/jtns/jtn002.html
 * Extended with additional assertion macros and test counting.
 */

#ifndef MINUNIT_H
#define MINUNIT_H

#include <stdio.h>
#include <string.h>

/* Test counters */
static int mu_tests_run = 0;
static int mu_tests_passed = 0;
static int mu_tests_failed = 0;

/* Current test name for error reporting */
static const char* mu_current_test = NULL;

/* Basic assertion - returns error message on failure */
#define mu_assert(message, test) do { \
    if (!(test)) { \
        return message; \
    } \
} while (0)

/* Assert equality for integers */
#define mu_assert_int_eq(expected, actual) do { \
    int _expected = (expected); \
    int _actual = (actual); \
    if (_expected != _actual) { \
        static char _msg[256]; \
        snprintf(_msg, sizeof(_msg), \
            "Expected %d but got %d", _expected, _actual); \
        return _msg; \
    } \
} while (0)

/* Run a test function */
#define mu_run_test(test) do { \
    mu_current_test = #test; \
    const char *message = test(); \
    mu_tests_run++; \
    if (message) { \
        mu_tests_failed++; \
        printf("  FAIL: %s\n", #test); \
        printf("        %s\n", message); \
    } else { \
        mu_tests_passed++; \
        printf("  PASS: %s\n", #test); \
    } \
} while (0)

/* Print test summary */
#define mu_print_summary() do { \
    printf("\n----------------------------------------\n"); \
    printf("Tests run: %d\n", mu_tests_run); \
    printf("Passed:    %d\n", mu_tests_passed); \
    printf("Failed:    %d\n", mu_tests_failed); \
    printf("----------------------------------------\n"); \
} while (0)

/* Return exit code based on test results */
#define mu_exit_code() (mu_tests_failed > 0 ? 1 : 0)

#endif /* MINUNIT_H */
