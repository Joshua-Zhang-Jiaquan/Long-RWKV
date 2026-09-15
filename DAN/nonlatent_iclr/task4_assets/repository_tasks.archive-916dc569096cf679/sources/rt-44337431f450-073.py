def get_parent_position(position: int) -> int:
    """
    heap helper function get the position of the parent of the current node

    >>> get_parent_position(1)
    0
    >>> get_parent_position(2)
    0
    """
    return (position - 1) // 2
