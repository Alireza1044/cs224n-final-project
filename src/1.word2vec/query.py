import numpy as np
import argparse
import os
from utils.treebank import StanfordSentiment
import matplotlib.pyplot as plt
from src import config

parser = argparse.ArgumentParser(description="1.word2vec arguments")
parser.add_argument("--char",
                    type=str,
                    default='michael')
parser.add_argument("-w",
                    type=list,
                    help="Query to process, should be an array of words e.g. ['hello','goodbye']")
args = parser.parse_args()
cls = args.char
dataset = StanfordSentiment(character=cls)
tokens = dataset.tokens()
print(f"training for: {cls}")

wordVectors = np.load(os.path.join(config.model_save_path, f"{cls}.word2vec.npy"))

visualizeWords = args.w

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
