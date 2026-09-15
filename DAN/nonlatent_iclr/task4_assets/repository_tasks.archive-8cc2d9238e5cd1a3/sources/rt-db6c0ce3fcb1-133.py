def gcd(x, y):
    """
    Function to find gcm (greatest common divisor) of two numbers
    :param x: first number
    :param y: second number
    :return: gcd of x and y
    """
    while y != 0:
        (x, y) = (y, x % y)
    return x
