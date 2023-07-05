#grid = [[3, 0, 6, 5, 0, 8, 4, 0, 0],
 #       [5, 2, 0, 0, 0, 0, 0, 0, 0],
  #      [0, 8, 7, 0, 0, 0, 0, 3, 1],
   #     [0, 0, 3, 0, 1, 0, 0, 8, 0],
    #    [9, 0, 0, 8, 6, 3, 0, 0, 5],
     #   [0, 5, 0, 0, 9, 0, 6, 0, 0],
        #[1, 3, 0, 0, 0, 0, 2, 5, 0],
      #  [0, 0, 0, 0, 0, 0, 0, 7, 4],
       # [0, 0, 5, 2, 0, 6, 3, 0, 0]]


def print_grid(grid):
    for row in grid:
        print(row)


def check_legal(grid, row, column, number):
    # Check row for duplicate number
    for x in range(9):
        if grid[row][x] == number:
            return False

    # Check column for duplicate number
    for x in range(9):
        if grid[x][column] == number:
            return False

    startRow = row - row % 3
    startColumn = column - column % 3
    for i in range(3):
        for j in range(3):
            if grid[i + startRow][j + startColumn] == number:
                return False
    return True

def solve_sudoku(grid, row, column):
    if row == 8 and column == 9:
        return True

    if column == len(grid[0]):
        column = 0
        row = row + 1

    if grid[row][column] > 0:
        return solve_sudoku(grid, row, column + 1)

    for n in range(1, len(grid) + 1, 1):
        if check_legal(grid, row, column, n):
            grid[row][column] = n
            if solve_sudoku(grid, row, column + 1):
                return True

        grid[row][column] = 0
    return False
