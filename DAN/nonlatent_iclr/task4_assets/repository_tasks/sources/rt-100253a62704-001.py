def get_check_digit(barcode: int) -> int:
    """
    Returns the last digit of barcode by excluding the last digit first
    and then computing to reach the actual last digit from the remaining
    12 digits.

    >>> get_check_digit(8718452538119)
    9
    >>> get_check_digit(87184523)
    5
    >>> get_check_digit(87193425381086)
    9
    >>> [get_check_digit(x) for x in range(0, 100, 10)]
    [0, 7, 4, 1, 8, 5, 2, 9, 6, 3]
    """
    barcode //= 10  # exclude the last digit
    checker = False
    s = 0

    # extract and check each digit
    while barcode != 0:
        mult = 1 if checker else 3
        s += mult * (barcode % 10)
        barcode //= 10
        checker = not checker

    return (10 - (s % 10)) % 10
