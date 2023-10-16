import json

with open("data/all.json") as user_file_all:
    file_contents_all = user_file_all.read()

data_all = json.loads(file_contents_all)

print(len(data_all))

