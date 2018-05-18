from random import randint
import numpy as np

def extract_pair(labels, nb_pair):
    pair_ml = []
    pair_cl = []
    b1 = True
    b2 = True
    n = labels.shape[0]
    while nb_pair > 0 or b1 or b2:
        i = randint(0, n-1)
        j = i
        while j == i:
            j = randint(0,n-1)
        if labels[i] == labels[j]:
            pair_ml.append(np.array([i,j]))
            b1 = False
        else:
            b2 = False
            pair_cl.append(np.array([i,j]))
        nb_pair-=1
    return pair_ml, pair_cl
