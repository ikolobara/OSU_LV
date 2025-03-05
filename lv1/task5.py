def task5():
    with open("SMSSpamCollection.txt") as f:
        lines = f.read().splitlines()
        lines = [item.replace('\t', ' ').strip() for item in lines]
            
    print(avg_word_count("ham", lines))
    print(avg_word_count("spam", lines))
    print (sum(1 for item in lines if item.startswith("spam") and item.endswith("!")))

def avg_word_count(starting_string: str, lines: list[str]) -> int: 
    return sum(len(item.split()) - 1 for item in lines if item.startswith(starting_string)) / sum(1 for item in lines if item.startswith(starting_string))
