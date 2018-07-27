
import matplotlib.pyplot as plt
import sys
import numpy as np


if __name__ == "__main__":
    plt.figure(figsize=(20, 10))
    datafile = open(sys.argv[1], "r")
    line = datafile.readline()
    tab = line.split(' ')
    tab = tab[:len(tab)-1]
    y = [float(x) for x in tab]
    x = [i for i in range(len(y))]
    plt.scatter(x,y)
    plt.xlabel('Index of document')
    plt.ylabel('Norm')
    plt.legend()
    plt.show()
    plt.savefig(sys.argv[2])
    print('taille %d ' % len(x))
