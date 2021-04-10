from hmm import BigramHMM, TrigramHMM, QuadgramHMM
from pinyin_hanzi import PinyinHanzi
import pickle


class PinyinIM:
    def __init__(self, table_dir, hmm="bigram"):
        self.pinyin_hanzi = PinyinHanzi(table_dir)
        self.hmm = self._get_hmm(hmm)

    def fit(self, corpus: list):
        self.hmm.fit(corpus)

    def predict(self, pinyins):
        O = list(map(self.pinyin_hanzi.encode_pinyin, pinyins))
        return list(map(self.pinyin_hanzi.decode_hanzi, self.hmm.predict(O)))

    def _get_hmm(self, hmm):
        if hmm == "bigram":
            return BigramHMM(self.pinyin_hanzi.N, self.pinyin_hanzi.M,
                             self.pinyin_hanzi.emission())
        elif hmm == "trigram":
            return TrigramHMM(self.pinyin_hanzi.N, self.pinyin_hanzi.M,
                              self.pinyin_hanzi.emission())
        elif hmm == "quadgram":
            return QuadgramHMM(self.pinyin_hanzi.N, self.pinyin_hanzi.M,
                              self.pinyin_hanzi.emission())

    def save(self, model_dir):
        with open(model_dir, "wb") as fout:
            pickle.dump(self, fout)

    def load(self, model_dir):
        with open(model_dir, 'rb') as fin:
            self.__dict__.update(pickle.load(fin).__dict__)
    
    def code_predict(self, pinyins):
        O = list(map(self.pinyin_hanzi.encode_pinyin, pinyins))
        return self.hmm.predict(O)