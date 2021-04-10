from pinyin_im import PinyinIM
import pickle
import argparse
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", type=str)
    parser.add_argument("--inputs", type=str)
    parser.add_argument("--train", type=str)
    parser.add_argument("--table_dir", type=str)
    parser.add_argument("--load", type=str, default=None)
    parser.add_argument("--save", type=str)
    parser.add_argument("--model",
                        choices=["bigram", "trigram", "quadgram"],
                        default="trigram")
    parser.add_argument("--lamb", nargs='+', default=None, type=float)

    return parser.parse_args()


def main():
    args = parse_args()
    print("Model: ", args.model)
    pinyin_im = PinyinIM(args.table_dir, args.model)

    with open(args.inputs, 'rb') as fin:
        X_test = pickle.load(fin)
    print(X_test[:2])
    with open(args.gt, "rb") as fin:
        y_test = pickle.load(fin)
    test_n = len(X_test)
    X_test, y_test = X_test[:test_n // 5000], y_test[:test_n // 5000]
    print(y_test[:2])

    if args.load is None:
        with open(args.train, 'rb') as fin:
            train_data = pickle.load(fin)
        print(train_data[:2])
        print("Training...")
        pinyin_im.fit(train_data)
        pinyin_im.save(args.save)
    else:
        print("Loading Model:", args.load)
        pinyin_im.load(args.load)
    if args.lamb is not None:
        print("Lambda: ", args.lamb)
        pinyin_im.hmm.set_lamb(args.lamb)

    print("Evaluating...")
    y_pred = pinyin_im.hmm.predict(X_test)

    sent_acc = 0
    word_acc = 0
    total_word = 0
    for pred, gold in tqdm(zip(y_pred, y_test)):
        total_word += len(pred)
        if pred == gold:
            sent_acc += 1
            word_acc += len(pred)
        else:
            print("Predict: ", pinyin_im.pinyin_hanzi.decode_hanzi(pred))
            print("Correct: ", pinyin_im.pinyin_hanzi.decode_hanzi(gold))
            for cp, cg in zip(pred, gold):
                if cp == cg:
                    word_acc += 1
    print("Sentence Accuracy: ", sent_acc / len(y_test))
    print("Word Accuracy: ", word_acc / total_word)


if __name__ == "__main__":
    main()
