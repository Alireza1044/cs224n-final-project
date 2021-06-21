import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader
from model import Model
from dataset import Dataset


def predict(dataset, model, text, next_words=100):
    model.eval()

    words = text.split(' ')
    state_h, state_c = model.init_state(len(words))

    for i in range(0, next_words):
        x = torch.tensor([[dataset.word_to_index[w] for w in words[i:]]])
        y_pred, (state_h, state_c) = model(x, (state_h, state_c))

        last_word_logits = y_pred[0][-1]
        p = torch.nn.functional.softmax(last_word_logits, dim=0).detach().numpy()
        word_index = np.random.choice(len(last_word_logits), p=p)
        words.append(dataset.index_to_word[word_index])

    return words


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default="hi")
    parser.add_argument('--char', type=str, default="michael")
    parser.add_argument('--length', type=int, default=5)
    parser.add_argument('--sequence-length', type=int, default=5)
    args = parser.parse_args()
    device = torch.device('cpu')

    print(f"Using: {device}")
    dataset = Dataset(args, device)
    model = Model(dataset, device)
    model.load_state_dict(torch.load(os.path.join("tmp", f"{args.char}.pth")))
    print(' '.join(predict(dataset, model, text=args.input, next_words=args.length)))
