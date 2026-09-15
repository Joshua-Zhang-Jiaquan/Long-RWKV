def gray_code(n: int) -> list[int]:
    """Return the n-bit Gray code sequence as a list of integers.

    Uses the reflection (mirror) construction:
        gray(i) = i ^ (i >> 1)

    >>> gray_code(2)
    [0, 1, 3, 2]
    >>> gray_code(3)
    [0, 1, 3, 2, 6, 7, 5, 4]
    """
    return [i ^ (i >> 1) for i in range(1 << n)]
