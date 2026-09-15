def solution(n: int = 1000) -> int:
    """
    Returns the sum of all the multiples of 3 or 5 below n.
    A straightforward pythonic solution using list comprehension.

    >>> solution(3)
    0
    >>> solution(4)
    3
    >>> solution(10)
    23
    >>> solution(600)
    83700
    """

    return sum(i for i in range(n) if i % 3 == 0 or i % 5 == 0)
