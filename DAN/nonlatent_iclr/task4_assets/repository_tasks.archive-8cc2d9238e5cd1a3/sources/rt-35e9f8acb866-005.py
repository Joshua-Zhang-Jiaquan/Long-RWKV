def combination_sum_iv_bottom_up(n: int, array: list[int], target: int) -> int:
    """
    Function checks the all possible combinations with using bottom up approach,
    and returns the count of possible combination in O(N^2) Time Complexity
    as we are using Dynamic programming array here.

    >>> combination_sum_iv_bottom_up(3, [1,2,5], 5)
    9
    """

    dp_array = [0] * (target + 1)
    dp_array[0] = 1

    for i in range(1, target + 1):
        for j in range(n):
            if i - array[j] >= 0:
                dp_array[i] += dp_array[i - array[j]]

    return dp_array[target]
