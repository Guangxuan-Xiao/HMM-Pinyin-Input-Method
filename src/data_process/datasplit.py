import pickle
import random
import os


def seed_everything(seed=42):

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


seed_everything()
news_dir = "../corpus/news/"
with open(news_dir + "sina_data.pkl", 'rb') as fin:
    data = pickle.load(fin)
n = len(data)
random.shuffle(data)
with open(news_dir + "sina_train.pkl", "wb") as handle:
    pickle.dump(data[:int(n * 0.8)], handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(news_dir + "sina_test.pkl", "wb") as handle:
    pickle.dump(data[int(n * 0.8):], handle, protocol=pickle.HIGHEST_PROTOCOL)
