def create_empty_board():
    """
    Create an empty 9x9 Sudoku board
    
    :return: 9x9 2D list filled with zeros
    """
    return [[0 for _ in range(9)] for _ in range(9)]
