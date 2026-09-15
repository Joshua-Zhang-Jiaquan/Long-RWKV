def create_canvas(size: int) -> list[list[bool]]:
    """
    Create a square canvas of given size filled with False (dead cells).

    Args:
        size: The dimension of the square canvas

    Returns:
        A size x size 2D list of boolean values, all initialized to False

    >>> canvas = create_canvas(3)
    >>> len(canvas)
    3
    >>> len(canvas[0])
    3
    >>> all(all(not cell for cell in row) for row in canvas)
    True
    >>> create_canvas(1)
    [[False]]
    >>> create_canvas(0)
    []
    """
    canvas = [[False for i in range(size)] for j in range(size)]
    return canvas
