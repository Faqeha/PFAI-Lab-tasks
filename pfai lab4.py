def is_safe(board, row, col, n):
    
    for i in range(col):
        if board[row][i] == 1:
            return False


    r, c = row, col
    while r >= 0 and c >= 0:
        if board[r][c] == 1:
            return False
        r -= 1
        c -= 1

    
    r, c = row, col
    while r < n and c >= 0:
        if board[r][c] == 1:
            return False
        r += 1
        c -= 1

    return True


def solve_n_queens(board, col, n):
    if col == n:  
        return True

    for row in range(n):
        if is_safe(board, row, col, n):
            board[row][col] = 1  
            if solve_n_queens(board, col + 1, n):
                return True

            board[row][col] = 0  

    return False


def print_board(board, n):
    print("\nSolution Board:\n")
    for i in range(n):
        for j in range(n):
            if board[i][j] == 1:
                print(" ", end="")
            else:
                print(" . ", end="")
        print()
    print("\n")



n = int(input("Enter value of N: "))
board = [[0]*n for _ in range(n)]

print("\nSolving N-Queens Problem...\n")

if solve_n_queens(board, 0, n):
    print("Solution Found!")
    print_board(board, n)
else:
    print("No Solution Exists!")
