def factorial(number):

    answer = 1

    if number == 0:
        return 1
    else:
        for num in range(1, number+1):
            answer = answer * num
    return answer
