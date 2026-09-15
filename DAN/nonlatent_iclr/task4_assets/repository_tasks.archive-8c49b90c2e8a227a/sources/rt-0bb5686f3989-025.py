def floyd(n):
    """
    Print the upper half of a diamond pattern with '*' characters.

    Args:
        n (int): Size of the pattern.

    Examples:
        >>> floyd(3)
        '  * \\n * * \\n* * * \\n'

        >>> floyd(5)
        '    * \\n   * * \\n  * * * \\n * * * * \\n* * * * * \\n'
    """
    result = ""
    for i in range(n):
        for _ in range(n - i - 1):  # printing spaces
            result += " "
        for _ in range(i + 1):  # printing stars
            result += "* "
        result += "\n"
    return result
