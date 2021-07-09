import numpy as np
import argparse
import os, sys, inspect
from sgd import *
from utils.treebank import StanfordSentiment
import matplotlib.pyplot as plt
import time
import os, sys, inspect
import argparse
sys.path.append("..")
import config

from word2vec import *
from sgd import *

parser = argparse.ArgumentParser(description="1.word2vec arguments")
parser.add_argument("--char",
                    type=str,
                    default='michael')
# parser.add_argument("-w",
#                     nargs='*',
#                     type=str,
#                     help="Query to process, should be an array of words, seperated by space e.g. hello goodbye")
args = parser.parse_args()
cls = args.char
dataset = StanfordSentiment(character=cls, path=config.data_path(args.char))
tokens = dataset.tokens()
print(f"training for: {cls}")
nWords = len(tokens)

# We are going to train 10-dimensional vectors for this assignment
dimVectors = 10

# Context size
C = 5

# Reset the random seed to make sure that everyone gets the same results
random.seed(31415)
np.random.seed(9265)

startTime = time.time()
wordVectors = np.concatenate(
    ((np.random.rand(nWords, dimVectors) - 0.5) /
     dimVectors, np.zeros((nWords, dimVectors))),
    axis=0)
wordVectors = sgd(
    lambda vec: word2vec_sgd_wrapper(skipgram, tokens, vec, dataset, C,
                                     negSamplingLossAndGradient),
    wordVectors, 0.3, 40000, os.path.join(config.model_save_path, f"{args.char}"), None, useSaved=True, PRINT_EVERY=10)


# visualizeWords = args.w
with open(config.data_path('michael'), 'r') as f:
    data = [l.strip() for l in f.readlines()]
with open(config.data_path('dwight'), 'r') as f:
    data2 = [l.strip() for l in f.readlines()]

visualizeWords = []
data = (' '.join(data)).split()
data2 = (' '.join(data2)).split()
data = list(set(data))
data2 = list(set(data2))

for d in data:
    if d in data2 and d in tokens:
        visualizeWords.append(d)

visualizeWords = sorted(visualizeWords)[:10]

visualizeIdx = [tokens[word] for word in visualizeWords]
visualizeVecs = wordVectors[visualizeIdx, :]
temp = (visualizeVecs - np.mean(visualizeVecs, axis=0))
covariance = 1.0 / len(visualizeIdx) * temp.T.dot(temp)
U, S, V = np.linalg.svd(covariance)
coord = temp.dot(U[:, 0:2])

for i in range(len(visualizeWords)):
    plt.text(coord[i, 0], coord[i, 1], visualizeWords[i],
             bbox=dict(facecolor='green', alpha=0.1))

plt.xlim((np.min(coord[:, 0]), np.max(coord[:, 0])))
plt.ylim((np.min(coord[:, 1]), np.max(coord[:, 1])))

plt.savefig(os.path.join(config.report_path, f"word_vectors_{cls}.png"))
