def longest_non_repeat_v2(string: str) -> int:
    """Find the length of the longest substring without repeating characters.

    Args:
        string: Input string to search.

    Returns:
        Length of the longest non-repeating substring.

    Examples:
        >>> longest_non_repeat_v2("abcabcbb")
        3
    """
    if string is None:
        return 0
    start, max_length = 0, 0
    used_char = {}
    for index, char in enumerate(string):
        if char in used_char and start <= used_char[char]:
            start = used_char[char] + 1
        else:
            max_length = max(max_length, index - start + 1)
        used_char[char] = index
    return max_length
