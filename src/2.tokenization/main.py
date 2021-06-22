import argparse
import os, sys, inspect
import sentencepiece as spm
from sklearn.model_selection import train_test_split

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import config


def load_data(char, flag):
    with open(config.data_path('michael_train'), 'r') as f:
        data = [d.strip() for d in f.readlines()]

    with open(config.data_path('dwight_train'), 'r') as f:
        data2 = [d.strip() for d in f.readlines()]

    if flag:
        data = data + data2
        return data
    elif char == 'michael':
        return data
    elif char == 'dwight':
        return data2


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--predict', action='store_true')
    parser.add_argument('-f', '--whole-data', action='store_true')
    parser.add_argument('--char', type=str, default="michael")
    parser.add_argument('--vocab-size', type=int, default=100)
    args = parser.parse_args()
    dataset = load_data(args.char, args.whole_data)
    if not args.predict:
        if args.whole_data:
            files = f"{config.data_path('michael_test')},{config.data_path('dwight_test')}"
        else:
            files = config.data_path(f"{args.char}_test")
        spm.SentencePieceTrainer.Train(
            f"--input={files} --vocab_size={args.vocab_size} --model_prefix={os.path.join(config.model_save_path, 'tokenization')}")
    else:
        spp = spm.SentencePieceProcessor()
        spp.load(os.path.join(config.model_save_path, 'tokenization.model'))
        unk = 0
        count = 0
        for data in dataset:
            ids = spp.EncodeAsIds(data)
            count += len(ids)
            unk += len([id for id in ids if ids == 0])
        print(count, unk)
        print(f"{(unk / float(count)) * 100}%")
