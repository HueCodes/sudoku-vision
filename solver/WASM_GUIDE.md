# Compiling the Sudoku Solver to WebAssembly

This guide walks through compiling the C solver to WASM so it can run in browsers.

**Estimated time:** 30-60 minutes for first setup, 5 minutes for subsequent builds.

---

## Prerequisites

### Install Emscripten SDK

```bash
# Clone the SDK
cd ~/Dev
git clone https://github.com/emscripten-core/emsdk.git
cd emsdk

# Install and activate latest version
./emsdk install latest
./emsdk activate latest

# Add to your shell (add to ~/.zshrc or ~/.bashrc for persistence)
source ~/Dev/emsdk/emsdk_env.sh
```

Verify installation:
```bash
emcc --version
# Should show: emcc (Emscripten gcc/clang-like replacement...) 3.x.x
```

---

## Step 1: Create WASM-Compatible Interface

The current code uses `main()` with interactive input. For WASM, we need exported functions that JavaScript can call.

Create `solver/src/wasm_api.c`:

```c
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
```

---

## Step 2: Update Header for WASM Exports

Ensure `solver/include/sudoku.h` exposes `is_safe`:

```c
#ifndef SUDOKU_H
#define SUDOKU_H

#define N 9
#define UNASSIGNED 0

int solve_sudoku(int grid[N][N]);
int is_safe(int grid[N][N], int row, int col, int num);
void print_grid(int grid[N][N]);

#endif
```

You may need to change `is_safe` from `static` to non-static in `sudoku.c` if it's currently static.

---

## Step 3: Create WASM Makefile

Create `solver/Makefile.wasm`:

```makefile
CC = emcc
SRC = src/sudoku.c src/wasm_api.c
OUT = ../web/sudoku.js

# Exported functions (the underscore prefix is required)
EXPORTS = _set_cell,_get_cell,_clear_grid,_solve,_is_valid

# Emscripten flags
EMFLAGS = -O3 \
          -s WASM=1 \
          -s EXPORTED_FUNCTIONS="[$(EXPORTS)]" \
          -s EXPORTED_RUNTIME_METHODS="[ccall,cwrap]" \
          -s MODULARIZE=1 \
          -s EXPORT_NAME="SudokuSolver" \
          -s ALLOW_MEMORY_GROWTH=0 \
          -s TOTAL_MEMORY=65536

.PHONY: all clean

all: $(OUT)

$(OUT): $(SRC)
	@mkdir -p ../web
	$(CC) $(EMFLAGS) -I include $(SRC) -o $(OUT)
	@echo "Built: $(OUT) and ../web/sudoku.wasm"

clean:
	rm -f ../web/sudoku.js ../web/sudoku.wasm
```

---

## Step 4: Build the WASM Module

```bash
cd ~/Dev/Projects/sudoku-vision/solver

# Make sure Emscripten is in PATH
source ~/Dev/emsdk/emsdk_env.sh

# Build
make -f Makefile.wasm
```

This produces:
- `web/sudoku.js` - JavaScript glue code
- `web/sudoku.wasm` - The compiled WebAssembly binary

---

## Step 5: Test in Browser

Create `web/index.html`:

```html
<!DOCTYPE html>
<html>
<head>
    <title>Sudoku Vision - Solver Test</title>
    <style>
        body { font-family: monospace; padding: 20px; }
        #output { white-space: pre; background: #f0f0f0; padding: 10px; }
    </style>
</head>
<body>
    <h1>Sudoku Solver WASM Test</h1>
    <button onclick="testSolver()">Run Test</button>
    <pre id="output"></pre>

    <script src="sudoku.js"></script>
    <script>
        let solver = null;

        // Initialize the WASM module
        SudokuSolver().then(module => {
            solver = {
                setCell: module.cwrap('set_cell', null, ['number', 'number', 'number']),
                getCell: module.cwrap('get_cell', 'number', ['number', 'number']),
                clearGrid: module.cwrap('clear_grid', null, []),
                solve: module.cwrap('solve', 'number', []),
                isValid: module.cwrap('is_valid', 'number', [])
            };
            log('WASM module loaded successfully');
        });

        function log(msg) {
            document.getElementById('output').textContent += msg + '\n';
        }

        function printGrid() {
            let s = '+-------+-------+-------+\n';
            for (let row = 0; row < 9; row++) {
                if (row > 0 && row % 3 === 0) {
                    s += '+-------+-------+-------+\n';
                }
                s += '|';
                for (let col = 0; col < 9; col++) {
                    if (col > 0 && col % 3 === 0) s += '|';
                    let val = solver.getCell(row, col);
                    s += ' ' + (val === 0 ? '.' : val);
                }
                s += ' |\n';
            }
            s += '+-------+-------+-------+';
            return s;
        }

        function testSolver() {
            if (!solver) {
                log('Module not loaded yet');
                return;
            }

            // Example puzzle (0 = empty)
            const puzzle = [
                [5,3,0, 0,7,0, 0,0,0],
                [6,0,0, 1,9,5, 0,0,0],
                [0,9,8, 0,0,0, 0,6,0],

                [8,0,0, 0,6,0, 0,0,3],
                [4,0,0, 8,0,3, 0,0,1],
                [7,0,0, 0,2,0, 0,0,6],

                [0,6,0, 0,0,0, 2,8,0],
                [0,0,0, 4,1,9, 0,0,5],
                [0,0,0, 0,8,0, 0,7,9]
            ];

            // Load puzzle
            solver.clearGrid();
            for (let row = 0; row < 9; row++) {
                for (let col = 0; col < 9; col++) {
                    solver.setCell(row, col, puzzle[row][col]);
                }
            }

            log('Input puzzle:');
            log(printGrid());
            log('');

            // Solve
            const start = performance.now();
            const solved = solver.solve();
            const elapsed = (performance.now() - start).toFixed(2);

            if (solved) {
                log(`Solved in ${elapsed}ms:`);
                log(printGrid());
            } else {
                log('No solution exists');
            }
        }
    </script>
</body>
</html>
```

---

## Step 6: Serve and Test

```bash
cd ~/Dev/Projects/sudoku-vision/web

# Simple HTTP server (WASM requires HTTP, not file://)
python3 -m http.server 8080
```

Open http://localhost:8080 in your browser and click "Run Test".

---

## Expected Output

```
WASM module loaded successfully
Input puzzle:
+-------+-------+-------+
| 5 3 . | . 7 . | . . . |
| 6 . . | 1 9 5 | . . . |
| . 9 8 | . . . | . 6 . |
+-------+-------+-------+
| 8 . . | . 6 . | . . 3 |
| 4 . . | 8 . 3 | . . 1 |
| 7 . . | . 2 . | . . 6 |
+-------+-------+-------+
| . 6 . | . . . | 2 8 . |
| . . . | 4 1 9 | . . 5 |
| . . . | . 8 . | . 7 9 |
+-------+-------+-------+

Solved in 0.15ms:
+-------+-------+-------+
| 5 3 4 | 6 7 8 | 9 1 2 |
| 6 7 2 | 1 9 5 | 3 4 8 |
| 1 9 8 | 3 4 2 | 5 6 7 |
+-------+-------+-------+
| 8 5 9 | 7 6 1 | 4 2 3 |
| 4 2 6 | 8 5 3 | 7 9 1 |
| 7 1 3 | 9 2 4 | 8 5 6 |
+-------+-------+-------+
| 9 6 1 | 5 3 7 | 2 8 4 |
| 2 8 7 | 4 1 9 | 6 3 5 |
| 3 4 5 | 2 8 6 | 1 7 9 |
+-------+-------+-------+
```

---

## Troubleshooting

**"emcc: command not found"**
- Run `source ~/Dev/emsdk/emsdk_env.sh` first

**"is_safe undefined"**
- Remove `static` keyword from `is_safe` function in `sudoku.c`

**CORS errors in browser**
- Must serve via HTTP, not file://. Use `python3 -m http.server`

**"out of memory"**
- Increase `TOTAL_MEMORY` in Makefile.wasm (try 131072 or higher)

---

## Next Steps

Once this works:
1. Integrate with the web app's camera capture
2. Connect digit recognition output to `set_cell` calls
3. Add Web Worker for non-blocking solve on complex puzzles
