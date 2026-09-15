def solution(n: int = 100) -> int:
    """
    Returns the difference between the sum of the squares of the first n
    natural numbers and the square of the sum.

    >>> solution(10)
    2640
    >>> solution(15)
    13160
    >>> solution(20)
    41230
    >>> solution(50)
    1582700
    """

    sum_cubes = (n * (n + 1) // 2) ** 2
    sum_squares = n * (n + 1) * (2 * n + 1) // 6
    return sum_cubes - sum_squares
