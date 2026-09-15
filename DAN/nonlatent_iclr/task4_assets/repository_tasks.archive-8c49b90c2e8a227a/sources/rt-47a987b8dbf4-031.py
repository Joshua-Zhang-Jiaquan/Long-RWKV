def get_distance(x: float, y: float, max_step: int) -> float:
    """
    Return the relative distance (= step/max_step) after which the complex number
    constituted by this x-y-pair diverges. Members of the Mandelbrot set do not
    diverge so their distance is 1.

    >>> get_distance(0, 0, 50)
    1.0
    >>> get_distance(0.5, 0.5, 50)
    0.061224489795918366
    >>> get_distance(2, 0, 50)
    0.0
    """
    a = x
    b = y
    for step in range(max_step):  # noqa: B007
        a_new = a * a - b * b + x
        b = 2 * a * b + y
        a = a_new

        # divergence happens for all complex number with an absolute value
        # greater than 4
        if a * a + b * b > 4:
            break
    return step / (max_step - 1)
