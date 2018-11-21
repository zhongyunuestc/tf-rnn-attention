from __future__ import print_function

import numpy as np
import tokenization as token

def zero_pad(X, seq_len):
    return np.array([x[:seq_len - 1] + [0] * max(seq_len - len(x), 1) for x in X])


def get_vocabulary_size(X):
    return max([max(x) for x in X]) + 1  # plus the 0th word


def fit_in_vocabulary(X, voc_size):
    return [[w for w in x if w < voc_size] for x in X]


def batch_generator(X, y, batch_size, model):
    """Primitive batch generator 
    """
    if model == "test" or model == "val":
        yield X,y
    size = X.shape[0]
    X_copy = X.copy()
    y_copy = y.copy()
    indices = np.arange(size)
    np.random.shuffle(indices)
    X_copy = X_copy[indices]
    y_copy = y_copy[indices]
    i = 0
    while True:
        if i + batch_size <= size:
            yield X_copy[i:i + batch_size], y_copy[i:i + batch_size]
            i += batch_size
        else:
            i = 0
            indices = np.arange(size)
            np.random.shuffle(indices)
            X_copy = X_copy[indices]
            y_copy = y_copy[indices]
            continue

full = token.FullTokenizer("vocab.txt", True)

def load_data(file_in):
    x = []
    y = []
    label = ["0","1"]
    for line in open(file_in):
        ls = line.strip().split("	")
        if len(ls) != 4 or ls[1] not in label:
            continue
        y.append(int(ls[1]))
        tt = full.tokenize(ls[3])
        tid = full.convert_tokens_to_ids(tt)
        x.append(tid)
    return np.array(x), np.array(y)

if __name__ == "__main__":
    # Test batch generator
    x,y = load_data("/home/zeuszhong/train.tsv")
    gen = batch_generator(x,y, 128)
    for _ in range(8):
        xx, yy = next(gen)
        #print(xx[0], yy)
        print (zero_pad(xx, 25))
        break
