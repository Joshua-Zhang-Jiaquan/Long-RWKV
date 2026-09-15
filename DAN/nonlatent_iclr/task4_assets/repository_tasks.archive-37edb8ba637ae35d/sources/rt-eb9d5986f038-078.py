def is_palindrome_traversal(s: str) -> bool:
    """
    Return True if s is a palindrome otherwise return False.

    >>> all(is_palindrome_traversal(key) == value for key, value in test_data.items())
    True
    """
    end = len(s) // 2
    n = len(s)

    # We need to traverse till half of the length of string
    # as we can get access of the i'th last element from
    # i'th index.
    # eg: [0,1,2,3,4,5] => 4th index can be accessed
    # with the help of 1st index (i==n-i-1)
    # where n is length of string
    return all(s[i] == s[n - i - 1] for i in range(end))
