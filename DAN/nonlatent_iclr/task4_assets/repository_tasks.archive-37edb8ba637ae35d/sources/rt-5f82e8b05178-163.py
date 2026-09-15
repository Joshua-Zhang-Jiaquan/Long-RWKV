def is_digit_cancelling(num: int, den: int) -> bool:
    return (
        num != den and num % 10 == den // 10 and (num // 10) / (den % 10) == num / den
    )
