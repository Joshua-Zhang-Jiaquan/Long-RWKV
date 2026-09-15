def reverse_floyd(n):
    """
    Print the lower half of a diamond pattern with '*' characters.

    Args:
        n (int): Size of the pattern.

    Examples:
        >>> reverse_floyd(3)
        '* * * \\n * * \\n  * \\n   '

        >>> reverse_floyd(5)
        '* * * * * \\n * * * * \\n  * * * \\n   * * \\n    * \\n     '
    """
    result = ""
    for i in range(n, 0, -1):
        for _ in range(i, 0, -1):  # printing stars
            result += "* "
        result += "\n"
        for _ in range(n - i + 1, 0, -1):  # printing spaces
            result += " "
    return result
