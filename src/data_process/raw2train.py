import numpy as np
import pickle
from tqdm import tqdm
table_dir = '../corpus/table/'
news_dir = '../corpus/news/'
data = np.load("../corpus/news/tsinghua_news.npy", allow_pickle=True)
hanzi2id = {}
with open(table_dir + "hanzi2id.txt") as fin:
    lines = fin.readlines()
for idx, line in enumerate(lines):
    line = line.strip().split(" ")
    hanzi2id[line[0]] = idx

pinyin2id = {}
with open(table_dir + "pinyin2id.txt") as fin:
    lines = fin.readlines()
for idx, line in enumerate(lines):
    line = line.strip().split(" ")
    pinyin2id[line[0]] = idx
id2pinyin = list(pinyin2id.keys())

hanzi2pinyin = np.load(table_dir + "hanzi2pinyin_id.npy")


def get_hanzi_id(hanzi, pinyin):
    i = 0
    while True:
        if hanzi + str(i) in hanzi2id:
            hanzi_id = hanzi2id[hanzi + str(i)]
        else:
            return hanzi2id[hanzi + "0"]
        if id2pinyin[hanzi2pinyin[hanzi_id]] == pinyin:
            return hanzi_id
        i += 1


def get_train_data():
    train_data = []
    total = data.shape[0]
    for line in tqdm(data, total=total):
        ids = []
        sentence = line[0]
        pinyin = line[1]
        for hanzi, pinyin in zip(sentence, pinyin):
            if hanzi + "0" not in hanzi2id:
                continue
            ids.append(get_hanzi_id(hanzi, pinyin))
        if len(ids) > 2:
            train_data.append(ids)

    with open(news_dir + "tsinghua_data.pkl", "wb") as handle:
        pickle.dump(train_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


get_train_data()