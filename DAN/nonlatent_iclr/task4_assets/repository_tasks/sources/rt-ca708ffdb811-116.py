def is_safe(row: int, col: int, rows: int, cols: int) -> bool:
    """
    Checking whether coordinate (row, col) is valid or not.

    >>> is_safe(0, 0, 5, 5)
    True
    >>> is_safe(-1,-1, 5, 5)
    False
    """
    return 0 <= row < rows and 0 <= col < cols
