def create_negative_cycle_graph():
    """
    Create a graph with a negative cycle for testing
    
    :return: dictionary representing a graph with negative cycle
    """
    return {
        'A': [('B', 1)],
        'B': [('C', -3)],
        'C': [('D', 2)],
        'D': [('B', -1)]
    }
