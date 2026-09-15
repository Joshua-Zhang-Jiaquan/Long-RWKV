def get_child_left_position(position: int) -> int:
    """
    heap helper function get the position of the left child of the current node

    >>> get_child_left_position(0)
    1
    """
    return (2 * position) + 1
