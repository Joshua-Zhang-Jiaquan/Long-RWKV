def create_sample_graph():
    """
    Create a sample weighted graph for testing
    
    :return: dictionary representing a weighted graph
    """
    return {
        'A': [('B', -1), ('C', 4)],
        'B': [('C', 3), ('D', 2), ('E', 2)],
        'C': [],
        'D': [('B', 1), ('C', 5)],
        'E': [('D', -3)]
    }
