def task4():
    word_count_dict = {}
    with open("song.txt") as f:
        for line in f:
            for word in line.lower().strip().replace(',', '').split(' '):
                if word in word_count_dict.keys():
                    word_count_dict[word] = word_count_dict.get(word) + 1
                else:
                    word_count_dict[word] = 1

    count = 0
    for key,value in word_count_dict.items():
        if value == 1:
            count += 1
            print(key)
        
    print(count)
