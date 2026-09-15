def is_palindrome_slice(s: str) -> bool:
    """
    Return True if s is a palindrome otherwise return False.

    >>> all(is_palindrome_slice(key) == value for key, value in test_data.items())
    True
    """
    return s == s[::-1]
