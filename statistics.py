import json

t = json.load(open(".data/squad/train-v1.1.json"))
d = json.load(open(".data/squad/dev-v1.1.json"))


def get_size(data):
    cnt = 0
    for data_topic in data["data"]:
        topic_title = data_topic["title"]
        for paragraph in data_topic["paragraphs"]:
            for question_and_answers in paragraph['qas']:
                cnt += 1
    return cnt


logging.info(f"Training data size is {get_size(t)}.")
logging.info(f"Dev data size is {get_size(d)}.")
