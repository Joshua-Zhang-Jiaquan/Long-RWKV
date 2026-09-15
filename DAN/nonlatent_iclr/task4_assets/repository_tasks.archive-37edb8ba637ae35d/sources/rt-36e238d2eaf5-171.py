def climb_stairs_optimized(steps: int) -> int:
    """Count distinct ways to climb n steps using constant space.

    Args:
        steps: Number of steps in the staircase (positive integer).

    Returns:
        Number of distinct ways to reach the top.

    Examples:
        >>> climb_stairs_optimized(2)
        2
        >>> climb_stairs_optimized(10)
        89
    """
    a_steps = b_steps = 1
    for _ in range(steps):
        a_steps, b_steps = b_steps, a_steps + b_steps
    return a_steps
