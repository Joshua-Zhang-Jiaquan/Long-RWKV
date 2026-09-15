def factorial(num: int) -> int:
    """Find the factorial of a given number n"""
    fact = 1
    for i in range(1, num + 1):
        fact *= i
    return fact
