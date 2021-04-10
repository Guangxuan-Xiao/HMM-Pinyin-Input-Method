from pinyin_im import PinyinIM
import pickle
import argparse
import time


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train", type=str, default=None, help="Training data directory."
    )
    parser.add_argument(
        "--load", type=str, default=None, help="Load trained model file."
    )
    parser.add_argument(
        "--save", type=str, default=None, help="Output trained model file."
    )
    parser.add_argument(
        "--input", type=str, default=None, help="Input pinyin text file."
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Output hanzi text file."
    )
    parser.add_argument(
        "--table_dir", type=str, help="Hanzi & Pinyin tables directory."
    )
    parser.add_argument(
        "--mode", choices=["test", "interact", "eval"], default="test", help="Run mode."
    )
    parser.add_argument(
        "--model",
        choices=["bigram", "trigram", "quadgram"],
        default="trigram",
        help="HMM model to use.",
    )
    parser.add_argument(
        "--lamb",
        nargs="+",
        default=None,
        type=float,
        help="List of HMM linear smoothing parameters.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    pinyin_im = PinyinIM(args.table_dir, args.model)
    if args.load is not None:
        print("Loading...")
        pinyin_im.load(args.load)
    if args.train is not None:
        with open(args.train, "rb") as fin:
            train_data = pickle.load(fin)
        print("Training...")
        pinyin_im.fit(train_data)
    if args.lamb is not None:
        print("Lambda:", args.lamb)
        pinyin_im.hmm.set_lamb(args.lamb)
    if args.save is not None:
        pinyin_im.save(args.save)

    if args.mode == "test":
        with open(args.input) as fin:
            pinyins = [
                line.strip().split(" ") for line in fin.readlines() if len(line) > 1
            ]
        print("Predicting...")
        outputs = pinyin_im.predict(pinyins)
        with open(args.output, "w+") as fout:
            for sentence in outputs:
                print(sentence, file=fout)

    elif args.mode == "interact":
        while True:
            pinyin = input("Pinyin: ")
            start = time.time()
            print(pinyin_im.predict([pinyin.split(" ")])[0])
            print("Time Cost:", time.time() - start, "s")


if __name__ == "__main__":
    main()
