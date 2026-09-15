def get_child_right_position(position: int) -> int:
    """
    heap helper function get the position of the right child of the current node

    >>> get_child_right_position(0)
    2
    """
    return (2 * position) + 2
