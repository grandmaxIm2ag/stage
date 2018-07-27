
import matplotlib.pyplot as plt
import sys
import numpy as np


if __name__ == "__main__":
    plt.figure(figsize=(20, 10))
    datafile = open(sys.argv[1], "r")
    line = datafile.readline()
    colors = ['b','g','r','c','m','y','w']
    iColor=0
    null_d = []
    d_in_trash = []
    null_d_in_trash = []
    nb_kw = []
    while line != "":
        print(line)
        tmp = []
        tmp=line.split(" ")
        nb_kw.append(int(tmp[0]))
        null_d.append(int(tmp[1]))
        d_in_trash.append(int(tmp[2]))
        null_d_in_trash.append(int(tmp[3]))
        line = datafile.readline()

    lines = []
    fig, ax = plt.subplots()
    fig.set_size_inches(4.5, 3.5)
    #ax.figure(num=None, figsize=(3.5, 1.5), dpi=10, facecolor='w', edgecolor='k')
    lines += ax.plot(nb_kw, null_d,colors[0], label='Document without keywords', linewidth=2)
    lines += ax.plot(nb_kw, d_in_trash,colors[1], label='Document in Trash Class', linewidth=2)
    lines += ax.plot(nb_kw, null_d_in_trash,colors[2], label='Document without keywords in Trash Class', linewidth=2)
    
    x = []
    for i in range(len(nb_kw)):
        x.append(str(nb_kw[i]))
        x.append('')

    fig.gca().set_xlim(1,5)
    fig.gca().xaxis.set_ticklabels(x)
    #fig.legend(loc='upper left', frameon=False)
    
    fig.legend()
    plt.xlabel('Number of keywords per classes')
    plt.ylabel('Number of document')
    fig.savefig(sys.argv[2], bbox_inches='tight')

    plt.close()
