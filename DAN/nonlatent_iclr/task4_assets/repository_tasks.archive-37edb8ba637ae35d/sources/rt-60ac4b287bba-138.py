def determinant(m00: float, m01: float, m10: float, m11: float) -> float:
    """
    Calculates the determinant of a 2x2 matrix:

    | m00  m01 |
    | m10  m11 |

    Args:
        m00 (float): Element in the first row, first column.
        m01 (float): Element in the first row, second column.
        m10 (float): Element in the second row, first column.
        m11 (float): Element in the second row, second column.

    Returns:
        float: The determinant of the matrix.

    Examples:
        # Determinant of the identity matrix (should be 1)
        >>> determinant(1, 0, 0, 1)
        1

        # Determinant of a matrix with two equal rows (should be 0)
        >>> determinant(1, 2, 1, 2)
        0

        # Determinant of a matrix with a negative determinant
        >>> determinant(1, 2, 3, 4)
        -2

        # Determinant of a matrix with larger numbers
        >>> determinant(10, 20, 30, 40)
        -200
    """
    return m00 * m11 - m10 * m01
