# Description: This file contains the functions to solve the sudoku puzzle

# Function to determine if a specified placement for a number is legal
# Input: grid, row, column, number
# Output: True if legal, False if illegal
def check_legal(grid, row, column, number):
    # Check the row for duplicate number
    for x in range(9):
        if grid[row][x] == number:
            return False

    # Check the column for duplicate number
    for x in range(9):
        if grid[x][column] == number:
            return False

    # Check the 3x3 grid for duplicate number
    start_row = row - row % 3
    start_column = column - column % 3
    for i in range(3):
        for j in range(3):
            if grid[i + start_row][j + start_column] == number:
                return False

    return True


# Recursive function to solve the sudoku puzzle
# Input: grid, row, column
# Output: True if solved, False if unsolvable
def solve_sudoku(grid, row, column):
    if row == 8 and column == 9:    # Base case, puzzle is solved
        return True

    if column == len(grid[0]):    # If we reach the end of a row, move to the next line
        column = 0
        row = row + 1

    if grid[row][column] > 0:   # If the current cell is already filled, move to the next cell
        return solve_sudoku(grid, row, column + 1)

    for n in range(1, len(grid) + 1, 1):    # Try all numbers from 1 to 9
        if check_legal(grid, row, column, n):   # If the number is legal, place it in the cell
            grid[row][column] = n
            if solve_sudoku(grid, row, column + 1):
                return True

        grid[row][column] = 0       # If the number is illegal, reset the cell and try again
    return False
