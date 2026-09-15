def egg_drop(n: int, k: int) -> int:
    """Find the minimum number of trials to identify the critical floor.

    Args:
        n: Number of eggs.
        k: Number of floors.

    Returns:
        Minimum number of trials in the worst case.

    Examples:
        >>> egg_drop(1, 2)
        2
        >>> egg_drop(2, 6)
        3
    """
    egg_floor = [[0 for _ in range(k + 1)] for _ in range(n + 1)]

    for i in range(1, n + 1):
        egg_floor[i][1] = 1
        egg_floor[i][0] = 0

    for j in range(1, k + 1):
        egg_floor[1][j] = j

    for i in range(2, n + 1):
        for j in range(2, k + 1):
            egg_floor[i][j] = _INT_MAX
            for x in range(1, j + 1):
                res = 1 + max(egg_floor[i - 1][x - 1], egg_floor[i][j - x])
                if res < egg_floor[i][j]:
                    egg_floor[i][j] = res

    return egg_floor[n][k]
