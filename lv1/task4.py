def task4():
    dict = {}
    with open("song.txt") as f:
        for line in f:
            for word in line.lower().strip().split(' '):
                if word in dict.keys():
                    dict[word] = dict.get(word) + 1
                else:
                    dict[word] = 1

    for k,v in dict.items():
        if v == 1:
            print(k)
            