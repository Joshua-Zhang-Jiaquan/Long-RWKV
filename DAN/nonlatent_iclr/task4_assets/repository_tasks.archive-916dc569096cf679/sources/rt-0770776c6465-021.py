def scaling(scaling_factor: float) -> list[list[float]]:
    """
    >>> scaling(5)
    [[5.0, 0.0], [0.0, 5.0]]
    """
    scaling_factor = float(scaling_factor)
    return [[scaling_factor * int(x == y) for x in range(2)] for y in range(2)]
