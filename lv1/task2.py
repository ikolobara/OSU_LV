def task2():
    try:
        number = float(input("Unesi broj u intervalu [0.0, 1.0]: "))
        if (number > 1.0 or number < 0.0):
            print("Broj izvan intervala.")
        elif (number >= 0.9):
            print("A")
        elif (number >= 0.8):
            print("B")
        elif (number >= 0.7):
            print("C")
        elif (number >= 0.6):
            print("D")
        else:
            print("F")
    except:
        print("Niste utipkali broj.")
    