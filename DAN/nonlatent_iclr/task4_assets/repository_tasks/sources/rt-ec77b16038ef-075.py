def solution(n: int = 4000000) -> int:
    """
    Returns the sum of all even fibonacci sequence elements that are lower
    or equal to n.

    >>> solution(10)
    10
    >>> solution(15)
    10
    >>> solution(2)
    2
    >>> solution(1)
    0
    >>> solution(34)
    44
    """

    i = 1
    j = 2
    total = 0
    while j <= n:
        if j % 2 == 0:
            total += j
        i, j = j, i + j

    return total
