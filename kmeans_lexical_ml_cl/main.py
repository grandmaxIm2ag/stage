#!/usr/bin/env python
#-*- coding: utf-8 -*-

import numpy as np
import random
import math
import autoencoder as au
import matplotlib.pyplot as plt
import sys
from sklearn.cluster import KMeans
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
import syntetic_data as synt
    
def load_syntetic_data():
    data = synt.syntetic(20,10,100, 3)
    return data.TFIDFvectorize(), data.labels_
#Main
if __name__ == "__main__":
    corpus, label = load_syntetic_data()
    autoencoder = au.Autoencoder(10, 12, 10, corpus, np.array([1]), np.array([[6,7],[9,1],[9,5]]), np.array([[6,7],[13,7]]))
    autoencoder.init_placeholder()
    autoencoder.init_weights()
    autoencoder.init_biases()
    autoencoder.init_layers()
    autoencoder.init_losses()
    autoencoder.train(100, 0.01, [0.25, 0.25, 0.25, 0.25])
    autoencoder.plot_loss()
