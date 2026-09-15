def is_palindrome(n: int | str) -> bool:
    """
    Return true if the input n is a palindrome.
    Otherwise return false. n can be an integer or a string.

    >>> is_palindrome(909)
    True
    >>> is_palindrome(908)
    False
    >>> is_palindrome('10101')
    True
    >>> is_palindrome('10111')
    False
    """
    n = str(n)
    return n == n[::-1]
