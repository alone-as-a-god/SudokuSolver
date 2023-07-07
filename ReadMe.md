# Sudoku Solver

Sudoku Solver is a tool to solve Sudokus by locating the Sudoku board in pictures and then filling out the empty cells.

It uses a form of backtracking to solve the puzzle.

## Usage
After running the program the user is presented with two options:
- use a locally saved image by entering its path
- use the image saved in the user's clipboard

After supplying either of those the program will use OpenCV and an OCR to find the Sudoku in the image and solve it.
If there is a solution, the program will display the solved sudoku in a new window.

## Installation
To install the program, simply clone the repository and install the requirements.
```bash
pip install -r requirements.txt
```
