def longest_palindromic_substring_DP(s):

    S = [[False for i in range(len(s))] for j in range(len(s))]

    max_palindrome = ""

    for i in range(len(s))[::-1]:
        for j in range(i, len(s)):
            # if j - 1 < 3, then there is one or two characters between these
            # two positions, implying that if s[i] == s[j]
            # then that small string is a palindrome
            # We check if the above cases is valid or i
            # they are larger, we use DP to check the substring
            # in between j and i
            S[i][j] = s[i] == s[j] and (j - i < 3 or S[i+1][j-1])
            if S[i][j] and j - i + 1 > len(max_palindrome):
                max_palindrome = s[i:j+1]

    return max_palindrome
