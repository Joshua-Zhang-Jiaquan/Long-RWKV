def is_right(x1: int, y1: int, x2: int, y2: int) -> bool:
    """
    Check if the triangle described by P(x1,y1), Q(x2,y2) and O(0,0) is right-angled.
    Note: this doesn't check if P and Q are equal, but that's handled by the use of
    itertools.combinations in the solution function.

    >>> is_right(0, 1, 2, 0)
    True
    >>> is_right(1, 0, 2, 2)
    False
    """
    if x1 == y1 == 0 or x2 == y2 == 0:
        return False
    a_square = x1 * x1 + y1 * y1
    b_square = x2 * x2 + y2 * y2
    c_square = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)
    return (
        a_square + b_square == c_square
        or a_square + c_square == b_square
        or b_square + c_square == a_square
    )
