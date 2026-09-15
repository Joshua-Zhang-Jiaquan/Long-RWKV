def find_smallest(arr):
    smallest = arr[0]
    smallest_index = 0
    
    for i in range(1, len(arr)):
        
        if arr[i] < smallest:
            smallest = arr[i]
            smallest_index = i
    return smallest_index
