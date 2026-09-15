def palindromic_string(input_string: str) -> str:
    """
    >>> palindromic_string('abbbaba')
    'abbba'
    >>> palindromic_string('ababa')
    'ababa'

    Manacher's algorithm which finds Longest palindromic Substring in linear time.

    1. first this convert input_string("xyx") into new_string("x|y|x") where odd
        positions are actual input characters.
    2. for each character in new_string it find corresponding length and
        store the length and left,right to store previously calculated info.
        (please look the explanation for details)

    3. return corresponding output_string by removing all "|"
    """
    max_length = 0

    # if input_string is "aba" than new_input_string become "a|b|a"
    new_input_string = ""
    output_string = ""

    # append each character + "|" in new_string for range(0, length-1)
    for i in input_string[: len(input_string) - 1]:
        new_input_string += i + "|"
    # append last character
    new_input_string += input_string[-1]

    # we will store the starting and ending of previous furthest ending palindromic
    # substring
    left, right = 0, 0

    # length[i] shows the length of palindromic substring with center i
    length = [1 for i in range(len(new_input_string))]

    # for each character in new_string find corresponding palindromic string
    start = 0
    for j in range(len(new_input_string)):
        k = 1 if j > right else min(length[left + right - j] // 2, right - j + 1)
        while (
            j - k >= 0
            and j + k < len(new_input_string)
            and new_input_string[k + j] == new_input_string[j - k]
        ):
            k += 1

        length[j] = 2 * k - 1

        # does this string is ending after the previously explored end (that is right) ?
        # if yes the update the new right to the last index of this
        if j + k - 1 > right:
            left = j - k + 1
            right = j + k - 1

        # update max_length and start position
        if max_length < length[j]:
            max_length = length[j]
            start = j

    # create that string
    s = new_input_string[start - max_length // 2 : start + max_length // 2 + 1]
    for i in s:
        if i != "|":
            output_string += i

    return output_string
