def solution(n: int = 1000) -> int:
    """
    Returns the sum of all the multiples of 3 or 5 below n.

    >>> solution(3)
    0
    >>> solution(4)
    3
    >>> solution(10)
    23
    >>> solution(600)
    83700
    """

    total = 0
    terms = (n - 1) // 3
    total += ((terms) * (6 + (terms - 1) * 3)) // 2  # total of an A.P.
    terms = (n - 1) // 5
    total += ((terms) * (10 + (terms - 1) * 5)) // 2
    terms = (n - 1) // 15
    total -= ((terms) * (30 + (terms - 1) * 15)) // 2
    return total
