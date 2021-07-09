import argparse
import os, sys, inspect
import sentencepiece as spm
from sklearn.model_selection import train_test_split

sys.path.append("..")
import config


def merge_data():
    with open(config.data_path("michael"), 'r') as f:
        michael = [l.strip() for l in f.readlines()]

    with open(config.data_path("dwight"), 'r') as f:
        dwight = [l.strip() for l in f.readlines()]
    whole = michael + dwight
    train, test = train_test_split(whole, train_size=0.8, test_size=0.2, shuffle=True)
    with open(config.data_path("whole_train"), "w") as f:
        for d in train:
            f.write(f"{d}\n")
    with open(config.data_path("whole_test"), "w") as f:
        for d in test:
            f.write(f"{d}\n")
    return config.data_path("whole_train"), config.data_path("whole_test")


def load_data(char, flag):
    with open(config.data_path('michael_test'), 'r') as f:
        data = [d.strip() for d in f.readlines()]

    with open(config.data_path('dwight_test'), 'r') as f:
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
    parser.add_argument('--vocab-size', type=int, default=[50, 150, 500, 1000, 1500])
    args = parser.parse_args()
        # if args.whole_data:
        #     files = f"{config.data_path('michael_train')},{config.data_path('dwight_train')}"
        # else:
        #     files = config.data_path(f"{args.char}_train")
    with open(os.path.join(config.log_path, "tokenization.log"), "w") as f:
        f.write("\n")
    for vocab_size in args.vocab_size:
        unk_avg = 0.0
        for iter in range(5):
            train_file, test_file = merge_data()
            model_name = os.path.join(config.model_save_path, f'{vocab_size}.tokenization')
            spm.SentencePieceTrainer.Train(
                f"--input={train_file} --vocab_size={vocab_size} --model_prefix={model_name} --model_type=unigram")
            spp = spm.SentencePieceProcessor()
            spp.load(f'{model_name}.model')
            unk = 0
            count = 0
            with open(test_file, 'r') as f:
                dataset = [d.strip() for d in f.readlines()]
            for data in dataset:
                ids = spp.EncodeAsIds(data)
                count += len(ids)
                unk += len([id for id in ids if id == 0])
            print()
            print(80 * "=")
            print(f"vocab size = {vocab_size}, iter {iter} unk% = {(unk / float(count)) * 100}%")
            print(80 * "=")
            print()
            with open(os.path.join(config.log_path, "tokenization.log"), "a") as f:
                f.write(f"vocab size = {vocab_size}, iter {iter} unk% = {(unk / float(count)) * 100}%\n")
            unk_avg += ((unk / float(count)) * 100)
        unk_avg /= 5
        with open(os.path.join(config.log_path, "tokenization.log"), "a") as f:
            f.write(f"vocab size = {vocab_size} unk% average = {unk_avg}\n")