def Linear_Search(Test_arr, val):
    index = 0
    for i in range(len(Test_arr)):
        if val > Test_arr[i]:
            index = i + 1
    return index
