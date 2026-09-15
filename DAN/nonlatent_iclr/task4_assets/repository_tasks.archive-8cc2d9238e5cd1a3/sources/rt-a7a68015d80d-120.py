def calculate_one_moving_point_and_one_stationary_line(point, velocity, line, offset):
    """
    Determine if the point moving at velocity will intersect the line.
    
    The line is positioned at offset. Given a moving point and line segment,
    determine if the point will ever intersect the line segment.
    
    .. caution::
        
        Points touching at the start are considered to be intersection. This 
        is because there is no way to get the "direction" of a stationary
        point like you can a line or polygon.
    
    :param point: the starting location of the point
    :type point: :class:`pygorithm.geometry.vector2.Vector2`
    :param velocity: the velocity of the point 
    :type velocity: :class:`pygorithm.geometry.vector2.Vector2`
    :param line: the geometry of the stationary line
    :type line: :class:`pygorithm.geometry.line2.Line2`
    :param offset: the offset of the line
    :type offset: :class:`pygorithm.geometry.vector2.Vector2`
    :returns: if the point will intersect the line, distance until intersection
    :rtype: bool, :class:`numbers.Number` or None
    """
    return False, -1
