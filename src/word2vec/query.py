import numpy as np
import argparse
from utils.treebank import StanfordSentiment
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="word2vec arguments")
parser.add_argument("-c",
                    type=int,
                    help="Name of the class: 1.Michael 2.Dwight",
                    choices=[1, 2],
                    default=1)
parser.add_argument("-w",
                    type=list,
                    help="Query tto process, should be an array of words e.g. ['hello','goodbye']")
args = parser.parse_args()
cls = 'michael' if args.c == 1 else 'dwight'
dataset = StanfordSentiment(character=cls)
tokens = dataset.tokens()
print(f"training for: {cls}")

wordVectors = np.load(f"../../models/{cls}.word2vec.npy")

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

plt.savefig(f"../../word_vectors_{cls}.png")
