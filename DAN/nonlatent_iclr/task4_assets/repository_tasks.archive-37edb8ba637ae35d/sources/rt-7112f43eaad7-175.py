def count_paths_dp(m: int, n: int) -> int:
    """Bottom-up DP — O(m*n) time, O(n) space.

    >>> count_paths_dp(3, 7)
    28
    >>> count_paths_dp(3, 3)
    6
    """
    row = [1] * n
    for _ in range(1, m):
        for j in range(1, n):
            row[j] += row[j - 1]
    return row[n - 1]
