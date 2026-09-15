def make_matrix(row_size: int = 4) -> list[list[int]]:
    """
    >>> make_matrix()
    [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
    >>> make_matrix(1)
    [[1]]
    >>> make_matrix(-2)
    [[1, 2], [3, 4]]
    >>> make_matrix(3)
    [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    >>> make_matrix() == make_matrix(4)
    True
    """
    row_size = abs(row_size) or 4
    return [[1 + x + y * row_size for x in range(row_size)] for y in range(row_size)]
