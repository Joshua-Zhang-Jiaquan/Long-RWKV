def largest_square_area_in_matrix_bottom_up_space_optimization(
    rows: int, cols: int, mat: list[list[int]]
) -> int:
    """
    Function updates the largest_square_area, using bottom up
    approach. with space optimization.

    >>> largest_square_area_in_matrix_bottom_up_space_optimization(2, 2, [[1,1], [1,1]])
    2
    >>> largest_square_area_in_matrix_bottom_up_space_optimization(2, 2, [[0,0], [0,0]])
    0
    """
    current_row = [0] * (cols + 1)
    next_row = [0] * (cols + 1)
    largest_square_area = 0
    for row in range(rows - 1, -1, -1):
        for col in range(cols - 1, -1, -1):
            right = current_row[col + 1]
            diagonal = next_row[col + 1]
            bottom = next_row[col]

            if mat[row][col] == 1:
                current_row[col] = 1 + min(right, diagonal, bottom)
                largest_square_area = max(current_row[col], largest_square_area)
            else:
                current_row[col] = 0
        next_row = current_row

    return largest_square_area
