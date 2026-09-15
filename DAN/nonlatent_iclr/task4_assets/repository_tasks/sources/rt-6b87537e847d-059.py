def find_seq(arr):
    seq = {}
    count = 0

    for num in arr:
        if num - 1 in seq:
            seq[num] = seq[num - 1] + 1
            count = max(count, seq[num])
        else:
            seq[num] = 1

    return count
