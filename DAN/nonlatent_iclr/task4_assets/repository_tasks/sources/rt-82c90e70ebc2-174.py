def solution(n: int = 20) -> int:
    """
    Solve by explicitly counting the paths with dynamic programming.

    >>> solution(6)
    924
    >>> solution(2)
    6
    >>> solution(1)
    2
    """

    counts = [[1 for _ in range(n + 1)] for _ in range(n + 1)]

    for i in range(1, n + 1):
        for j in range(1, n + 1):
            counts[i][j] = counts[i - 1][j] + counts[i][j - 1]

    return counts[n][n]
