def max_subarray(array: list[int]) -> int:
    """Find the maximum sum of a contiguous subarray using Kadane's algorithm.

    Args:
        array: List of integers (containing at least one number).

    Returns:
        Largest sum among all contiguous subarrays.

    Examples:
        >>> max_subarray([1, 2, -3, 4, 5, -7, 23])
        25
    """
    max_so_far = max_now = array[0]
    for i in range(1, len(array)):
        max_now = max(array[i], max_now + array[i])
        max_so_far = max(max_so_far, max_now)
    return max_so_far
