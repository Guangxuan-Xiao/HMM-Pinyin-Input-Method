from pinyin_hanzi import PinyinHanzi
import pickle
from tqdm import tqdm
table_dir = '../corpus/table/'
news_dir = '../corpus/news/'
encoder = PinyinHanzi(table_dir)
with open(news_dir + "sina_test.pkl", 'rb') as fin:
    data = pickle.load(fin)
inputs = []
for sent in tqdm(data):
    inputs.append(encoder.convert_hanzi_to_pinyin(sent))
with open(news_dir + "sina_inputs.pkl", "wb") as handle:
    pickle.dump(inputs, handle, protocol=pickle.HIGHEST_PROTOCOL)
