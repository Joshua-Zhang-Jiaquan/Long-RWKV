def split_and_add(number: int) -> int:
    """Split number digits and add them."""
    sum_of_digits = 0
    while number > 0:
        last_digit = number % 10
        sum_of_digits += last_digit
        number = number // 10  # Removing the last_digit from the given number
    return sum_of_digits
