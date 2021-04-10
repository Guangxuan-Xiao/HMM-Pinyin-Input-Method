import pandas as pd
import os
from pypinyin import lazy_pinyin
from tqdm import tqdm
import re
import numpy as np
import pickle
table_dir = '../corpus/table/'


def split_text(text):
    pattern = r',|\.|/|;|\'|`|\[|\]|<|>|\?|:|"|\{|\}|\~|!|@|#|\$|%|\^|&|\(|\)|-|=|\_|\+|，|。|、|；|‘|’|【|】|·|！|:|：| |…|（|）|[0-9]+|[a-zA-Z]+|《|》|“|”|？|·|~|——|-'
    return [
        sentence for sentence in re.split(pattern, text) if len(sentence) > 1
    ]


def parse_sina(news_dir):
    all_data = np.array([]).reshape((0, 2))
    for file in tqdm(os.listdir(news_dir)):
        if ".txt" not in file:
            continue
        print(file)
        data = np.array([]).reshape((0, 2))
        news_df = pd.read_json(os.path.join(news_dir, file), lines=True)
        for text in news_df.html:
            sentences = split_text(text)
            pinyins = list(map(lazy_pinyin, sentences))
            data = np.vstack((data, np.array([sentences, pinyins]).T))
        all_data = np.vstack((all_data, data))
    np.save(os.path.join(news_dir, "sina_news"), all_data)


def parse_table(table_dir):
    pinyin_table = {}
    hanzi2pinyin = {}
    pinyin2hanzi = {}
    with open(os.path.join(table_dir, "pinyin_table.txt")) as fin:
        lines = fin.readlines()
    for line in lines:
        l = line.strip().split(" ")
        pinyin_table[l[0]] = l[1:]
    for k, v in pinyin_table.items():
        for c in v:
            if c not in hanzi2pinyin:
                hanzi2pinyin[c] = [k]
            else:
                hanzi2pinyin[c].append(k)
    fout = open(os.path.join(table_dir, "hanzi2pinyin.csv"), "w+")
    for k, v in hanzi2pinyin.items():
        for idx, pinyin in enumerate(v):
            print(k + str(idx) + "," + pinyin, file=fout)
            if pinyin not in pinyin2hanzi:
                pinyin2hanzi[pinyin] = [k + str(idx)]
            else:
                pinyin2hanzi[pinyin].append(k + str(idx))
    fout.close()
    fout = open(os.path.join(table_dir, "pinyin2hanzi.csv"), "w+")
    for k, v in pinyin2hanzi.items():
        print(k, file=fout, end=",")
        for c in v:
            print(c, file=fout, end=" ")
        print("", file=fout)
    fout.close()


def parse_tsinghua(news_dir):
    all_data = np.array([]).reshape((0, 2))
    tnews_df = pd.read_json(news_dir + "news.json", lines=True)
    news_list = [tnews_df.iloc[0, i] for i in range(4)]
    for news in tqdm(news_list, total=4):
        for piece in tqdm(news):
            if piece is None:
                continue
            data = np.array([]).reshape((0, 2))
            for text in ['title', 'content']:
                sentences = split_text(piece[text])
                pinyins = list(map(lazy_pinyin, sentences))
                data = np.vstack((data, np.array([sentences, pinyins]).T))
            all_data = np.vstack((all_data, data))
    np.save(os.path.join(news_dir, "tsinghua_news"), all_data)


if __name__ == "__main__":
    # parse_sina("../corpus/news/")
    # parse_table("../corpus/table")
    parse_tsinghua("../corpus/news/")