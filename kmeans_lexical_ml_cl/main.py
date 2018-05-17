import numpy as np
import random
import math
import autoencoder as au
import matplotlib.pyplot as plt
import sys


def load_data(filename):
    datafile = open(filename,'r')
    l = datafile.readline()
    mat = []
    for i in range (0, int(l)):
        str = datafile.readline()
        tmp = []
        tmp = str[:-1 ].split(":")
        tmp2 = []
        for j in range(len(tmp)):
            tmp2.append(float(tmp[j]))
        mat.append(tmp)
    return mat

if __name__ == "__main__":
    print str(load_data('base'))
