import csv

def convert_data():

    file = open("data/TinyStoriesV2-GPT4-valid.txt",mode='r',encoding="utf8")
    new_file = open("data/TinyStories/TinyStoriesV2-valid.csv", mode='x', encoding="utf8", newline='')
    csv_writer = csv.writer(new_file)
    csv_writer.writerow(["text"])
    current_text = ''
    i = 0
    j = 0
    while True:
        line = file.readline()
        if not line:
            break
        print(f"\r i: {i} , j: {j}", end="")
        j += 1
        if line == '<|endoftext|>\n':
            j = 0
            i += 1
            csv_writer.writerow([current_text + '<|endoftext|>'])
            current_text = ""
        else:
            current_text += line[:-1]
    file.close()
    new_file.close()

if __name__ == '__main__':
    convert_data()