import numpy as np
import os, sys
from sklearn.model_selection import train_test_split

sys.path.append("..")
import config

if __name__ == '__main__':
    with open("../data/clean/cleaned_broken_sentences/michael.txt", 'r') as f:
        michael = [l.strip() for l in f.readlines()]

    with open("../data/clean/cleaned_broken_sentences/dwight.txt", 'r') as f:
        dwight = [l.strip() for l in f.readlines()]
    whole = michael + dwight
    train, test = train_test_split(train_size=0.8, test_size=0.2, shuffle=True)