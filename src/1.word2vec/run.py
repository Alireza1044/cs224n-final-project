#!/usr/bin/env python

import random
import numpy as np
from utils.treebank import StanfordSentiment
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
import time
import os, sys, inspect
import argparse

sys.path.append("..")
import config

from word2vec import *
from sgd import *

# Check Python Version
assert sys.version_info[0] == 3
assert sys.version_info[1] >= 5

# Reset the random seed to make sure that everyone gets the same results
parser = argparse.ArgumentParser(description="1.word2vec arguments")
parser.add_argument('--char', type=str, default="michael")
parser.add_argument('--all', action="store_true")
parser.add_argument('--predict', action="store_true")
args = parser.parse_args()
cls = args.char
print(f"training for: {cls}")
random.seed(314)
dataset = StanfordSentiment(character=cls, path=config.data_path(cls))
tokens = dataset.tokens()
nWords = len(tokens)

# We are going to train 10-dimensional vectors for this assignment
dimVectors = 10

# Context size
C = 5

# Reset the random seed to make sure that everyone gets the same results
random.seed(31415)
np.random.seed(9265)
c = 'all' if args.all else args.char
startTime = time.time()
wordVectors = np.concatenate(
    ((np.random.rand(nWords, dimVectors) - 0.5) /
     dimVectors, np.zeros((nWords, dimVectors))),
    axis=0)
wordVectors = sgd(
    lambda vec: word2vec_sgd_wrapper(skipgram, tokens, vec, dataset, C,
                                     negSamplingLossAndGradient),
    wordVectors, 0.3, 40000, os.path.join("..", "..", config.model_save_path, f"{c}"), None, useSaved=args.predict, PRINT_EVERY=10)
# Note that normalization is not called here. This is not a bug,
# normalizing during training loses the notion of length.

print("sanity check: cost at convergence should be around or below 10")
print("training took %d seconds" % (time.time() - startTime))

# concatenate the input and output word vectors
wordVectors = np.concatenate(
    (wordVectors[:nWords, :], wordVectors[nWords:, :]),
    axis=0)
# np.save(os.path.join("..", "..", config.model_save_path, f"{cls}.word2vec.npy"), wordVectors)
