def score_function(
    source_char: str,
    target_char: str,
    match: int = 1,
    mismatch: int = -1,
    gap: int = -2,
) -> int:
    """
    Calculate the score for a character pair based on whether they match or mismatch.
    Returns 1 if the characters match, -1 if they mismatch, and -2 if either of the
    characters is a gap.
    >>> score_function('A', 'A')
    1
    >>> score_function('A', 'C')
    -1
    >>> score_function('-', 'A')
    -2
    >>> score_function('A', '-')
    -2
    >>> score_function('-', '-')
    -2
    """
    if "-" in (source_char, target_char):
        return gap
    return match if source_char == target_char else mismatch
