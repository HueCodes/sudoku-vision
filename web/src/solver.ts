/**
 * WASM Sudoku Solver wrapper.
 * Interfaces with the compiled C solver.
 */

declare const SudokuSolver: () => Promise<any>;

export interface SolverInterface {
  setCell: (row: number, col: number, value: number) => void;
  getCell: (row: number, col: number) => number;
  clearGrid: () => void;
  solve: () => boolean;
  isValid: () => boolean;
}

let solverInstance: SolverInterface | null = null;
let solverModule: any = null;

/**
 * Initialize the WASM solver module.
 */
export async function initSolver(): Promise<void> {
  if (solverInstance) return;

  // Load the solver script dynamically
  await new Promise<void>((resolve, reject) => {
    if (typeof SudokuSolver !== 'undefined') {
      resolve();
      return;
    }

    const script = document.createElement('script');
    script.src = '/sudoku.js';
    script.onload = () => resolve();
    script.onerror = () => reject(new Error('Failed to load solver.js'));
    document.head.appendChild(script);
  });

  solverModule = await SudokuSolver();

  solverInstance = {
    setCell: solverModule.cwrap('set_cell', null, ['number', 'number', 'number']),
    getCell: solverModule.cwrap('get_cell', 'number', ['number', 'number']),
    clearGrid: solverModule.cwrap('clear_grid', null, []),
    solve: solverModule.cwrap('solve', 'number', []),
    isValid: solverModule.cwrap('is_valid', 'number', []),
  };
}

/**
 * Solve a sudoku puzzle.
 * @param grid - 2D array (9x9) with 0 for empty cells
 * @returns Solved grid or null if unsolvable
 */
export async function solve(grid: number[][]): Promise<number[][] | null> {
  if (!solverInstance) {
    await initSolver();
  }

  const solver = solverInstance!;

  // Clear and load puzzle
  solver.clearGrid();
  for (let row = 0; row < 9; row++) {
    for (let col = 0; col < 9; col++) {
      solver.setCell(row, col, grid[row][col]);
    }
  }

  // Validate
  if (!solver.isValid()) {
    return null;
  }

  // Solve
  const solved = solver.solve();
  if (!solved) {
    return null;
  }

  // Extract solution
  const solution: number[][] = [];
  for (let row = 0; row < 9; row++) {
    solution[row] = [];
    for (let col = 0; col < 9; col++) {
      solution[row][col] = solver.getCell(row, col);
    }
  }

  return solution;
}

/**
 * Validate a puzzle (check for duplicate numbers).
 */
export async function validate(grid: number[][]): Promise<boolean> {
  if (!solverInstance) {
    await initSolver();
  }

  const solver = solverInstance!;

  solver.clearGrid();
  for (let row = 0; row < 9; row++) {
    for (let col = 0; col < 9; col++) {
      solver.setCell(row, col, grid[row][col]);
    }
  }

  return solver.isValid();
}
