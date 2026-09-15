def search(target: int, prime_list: list) -> bool:
    """
    function to search a number in a list using Binary Search.
    >>> search(3, [1, 2, 3])
    True
    >>> search(4, [1, 2, 3])
    False
    >>> search(101, list(range(-100, 100)))
    False
    """

    left, right = 0, len(prime_list) - 1
    while left <= right:
        middle = (left + right) // 2
        if prime_list[middle] == target:
            return True
        elif prime_list[middle] < target:
            left = middle + 1
        else:
            right = middle - 1

    return False
