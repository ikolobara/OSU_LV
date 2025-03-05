def task3():
    numbers = []
    while True:
        
        user_input = input("Unesi broj: ")
        if (user_input == "Done"):
                break
        try:
            numbers.append(int(user_input))
        except:
            print("Niste unijeli broj.")
    print(len(numbers))
    print(sum(numbers) / len(numbers))
    print(min(numbers))
    print(max(numbers))
    numbers.sort()
    print(numbers)
