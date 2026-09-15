def super_digit(n_str: str, repetitions: int) -> int:
    """
    Computes the super digit of a number formed by concatenating
    n_str repeated times.

    Parameters:
    n_str (str): The string representation of the integer.
    repetitions (int): The number of times to concatenate n_str.

    Returns:
    int: The super digit of the concatenated number.

    >>> super_digit("148", 3)
    3
    >>> super_digit("9875", 4)
    8
    >>> super_digit("123", 3)
    9
    """

    # Calculate the initial sum of the digits in n_str
    digit_sum = sum(int(digit) for digit in n_str)

    # Multiply the sum by repetitions
    total_sum = digit_sum * repetitions

    # Recursive function to find the super digit
    while total_sum >= 10:
        total_sum = sum(int(digit) for digit in str(total_sum))

    return total_sum
