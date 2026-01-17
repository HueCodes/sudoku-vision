# Makefile for C Sudoku Solver

# Compiler
CC = gcc

# Compiler flags
CFLAGS = -Wall -Wextra -std=c99 -pedantic -g

# Directories
SRCDIR = src
INCDIR = include
OBJDIR = obj
TESTDIR = tests

# Target executable
TARGET = sudoku_solver
TEST_TARGET = test_runner

# Source files (exclude wasm_api.c for native build)
SOURCES = $(filter-out $(SRCDIR)/wasm_api.c, $(wildcard $(SRCDIR)/*.c))
OBJECTS = $(SOURCES:$(SRCDIR)/%.c=$(OBJDIR)/%.o)

# Test files
TEST_SOURCES = $(TESTDIR)/test_solver.c
# Solver source without main.c for testing
SOLVER_SRC = $(SRCDIR)/sudoku.c
TEST_OBJECTS = $(OBJDIR)/sudoku.o

# Default target
all: $(TARGET)

# Create object directory if it doesn't exist
$(OBJDIR):
	mkdir -p $(OBJDIR)

# Link object files to create executable
$(TARGET): $(OBJECTS)
	$(CC) $(OBJECTS) -o $(TARGET)
	@echo "Build complete! Run with: ./$(TARGET)"

# Compile source files to object files
$(OBJDIR)/%.o: $(SRCDIR)/%.c | $(OBJDIR)
	$(CC) $(CFLAGS) -I$(INCDIR) -c $< -o $@

# Build and run tests
test: $(TEST_TARGET)
	./$(TEST_TARGET)

# Build test runner
$(TEST_TARGET): $(TEST_OBJECTS) $(TEST_SOURCES) | $(OBJDIR)
	$(CC) $(CFLAGS) -I$(INCDIR) $(TEST_SOURCES) $(TEST_OBJECTS) -o $(TEST_TARGET)

# Clean build files
clean:
	rm -rf $(OBJDIR) $(TARGET) $(TEST_TARGET)
	@echo "Clean complete."

# Run the program
run: $(TARGET)
	./$(TARGET)

# Debug build
debug: CFLAGS += -DDEBUG
debug: $(TARGET)

# Release build
release: CFLAGS += -O2 -DNDEBUG
release: clean $(TARGET)

# Show help
help:
	@echo "Available targets:"
	@echo "  all      - Build the sudoku solver (default)"
	@echo "  test     - Build and run unit tests"
	@echo "  clean    - Remove build files"
	@echo "  run      - Build and run the program"
	@echo "  debug    - Build with debug symbols"
	@echo "  release  - Build optimized release version"
	@echo "  help     - Show this help message"

.PHONY: all clean run debug release help test