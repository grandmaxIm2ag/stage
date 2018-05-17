#!/usr/bin/env python
#-*- coding: utf-8 -*-

import numpy as np
import random
import math
#import autoencoder as au
import matplotlib.pyplot as plt
import sys
from sklearn.cluster import KMeans

def load_data(filename):
    corpus = []
    label = []
    N = 0
    with open(filename) as openfileobject:
        for line in openfileobject:
            line2 = line.split(' ')
            label.append(int(line2[0]))
            doc = np.zeros(30000)
            for elem in line2[1:-1]:
                p = elem.split(':')
                doc[int(p[0])] = float(p[1])
            corpus.append(doc)
            N+=1    
    return label, np.array(corpus), N

if __name__ == "__main__":
    label, corpus,N = load_data('base')
    kmeans = KMeans(n_clusters=25, random_state=0).fit(corpus)
    print "label kmeans : \n"+str(kmeans.labels_)
    print "label vrais  : \n"+str(label)
    #autoencoder = au.Autoencoder(15000, 20000, 20000, corpus, np.array([1, 2, 8, 75, 432, 651, 980, 1500, 2381, 5012]))
    #autoencoder.init_placeholder()
    #autoencoder.init_weights()
    #autoencoder.init_biases()
    #autoencoder.init_layers()
    #autoencoder.init_losses()
    #autoencoder.train(100, 0.01, [0.5, 0.5])
    #autoencoder.plot_loss()
