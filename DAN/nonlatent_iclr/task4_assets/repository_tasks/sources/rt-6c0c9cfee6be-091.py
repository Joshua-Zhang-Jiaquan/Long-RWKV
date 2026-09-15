def longest_palindromic_substring_expansion(s):
    max_palindrome = ""

    for i in range(len(s) * 2 - 1):
        if i % 2 == 0:
            # This is when you are "on" an actual character
            # o = offset, ind = current character
            o = 0
            ind = i // 2
            while ind + o < len(s) and ind - o >= 0:
                if(s[ind + o] != s[ind - o]):
                    break
                if ind + o - (ind - o) + 1 > len(max_palindrome):
                    max_palindrome = s[ind-o:ind+o + 1]
                o += 1
        else:
            # This is when you are "in the middle of" two characters
            # o = offset, sind = start char, eind = end char
            o = 0
            sind = i // 2
            eind = i // 2 + 1
            while sind - o >= 0 and eind + o < len(s):
                if(s[sind - o] != s[eind + o]):
                    break
                if eind + o - (sind - o) + 1 > len(max_palindrome):
                    max_palindrome = s[sind - o:eind + o + 1]
                o += 1

    return max_palindrome
