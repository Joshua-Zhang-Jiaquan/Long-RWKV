def get_squares(n: int) -> list[int]:
    """
    >>> get_squares(0)
    []
    >>> get_squares(1)
    [0]
    >>> get_squares(2)
    [0, 1]
    >>> get_squares(3)
    [0, 1, 4]
    >>> get_squares(4)
    [0, 1, 4, 9]
    """
    return [number * number for number in range(n)]
