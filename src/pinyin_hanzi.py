import numpy as np
import os
import pickle


class PinyinHanzi:
    def __init__(self, table_dir):
        self.pinyin2id, self.id2pinyin = self._read_pinyin2id(table_dir)
        self.hanzi2id, self.id2hanzi = self._read_hanzi2id(table_dir)
        self.hanzi2pinyin = self._read_hanzi2pinyin(table_dir)
        self.pinyin2hanzi = self._read_pinyin2hanzi(table_dir)
        self.N = len(self.id2hanzi)
        self.M = len(self.id2pinyin)

    def _read_pinyin2id(self, table_dir):
        pinyin2id = {}
        id2pinyin = []
        with open(table_dir + "pinyin2id.txt") as fin:
            lines = fin.readlines()
        for idx, line in enumerate(lines):
            line = line.strip().split(" ")
            pinyin2id[line[0]] = idx
            id2pinyin.append(line[0])
        return pinyin2id, id2pinyin

    def _read_hanzi2id(self, table_dir):
        hanzi2id = {}
        id2hanzi = []
        with open(table_dir + "hanzi2id.txt") as fin:
            lines = fin.readlines()
        for idx, line in enumerate(lines):
            line = line.strip().split(" ")
            hanzi2id[line[0]] = idx
            id2hanzi.append(line[0])
        return hanzi2id, id2hanzi

    def _read_pinyin2hanzi(self, table_dir):
        with open(os.path.join(table_dir, 'pinyin2hanzi_id.pkl'),
                  'rb') as handle:
            pinyin2hanzi = pickle.load(handle)
        return pinyin2hanzi

    def _read_hanzi2pinyin(self, table_dir):
        return np.load(os.path.join(table_dir, "hanzi2pinyin_id.npy"))

    def emission(self):
        return self.pinyin2hanzi

    def decode_hanzi(self, hanzi_ids):
        return "".join([self.id2hanzi[c][0] for c in hanzi_ids])

    def encode_pinyin(self, pinyins):
        return [self.pinyin2id[p] for p in pinyins if p in self.pinyin2id]

    def convert_hanzi_to_pinyin(self, hanzi_ids):
        pinyins = []
        for hanzi in hanzi_ids:
            pinyins.append(self.hanzi2pinyin[hanzi])
        return pinyins
