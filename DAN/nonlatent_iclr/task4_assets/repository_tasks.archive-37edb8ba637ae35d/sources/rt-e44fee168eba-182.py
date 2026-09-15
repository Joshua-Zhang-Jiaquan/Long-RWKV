def is_palindrome(n: int) -> bool:
    """
    Returns True if a number is palindrome.
    >>> is_palindrome(12567321)
    False
    >>> is_palindrome(1221)
    True
    >>> is_palindrome(9876789)
    True
    """
    return str(n) == str(n)[::-1]
