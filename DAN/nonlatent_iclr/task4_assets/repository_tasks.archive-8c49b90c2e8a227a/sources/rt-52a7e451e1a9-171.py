def create_sample_graph():
    """
    Create a sample weighted graph for testing
    
    :return: dictionary representing a weighted graph
    """
    return {
        'A': [('B', 3), ('D', 7)],
        'B': [('A', 8), ('C', 2)],
        'C': [('A', 5), ('D', 1)],
        'D': [('A', 2)]
    }
