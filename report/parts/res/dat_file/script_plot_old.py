
import matplotlib.pyplot as plt
import sys
import numpy as np


if __name__ == "__main__":
    plt.figure(figsize=(20, 10))
    datafile = open(sys.argv[1], "r")
    line = datafile.readline()
    colors = ['b','g','r','c','m','y','w']
    iColor=0
    lex_discr = []
    lex_discr_add = []
    nb_kw = []
    dkm = []
    while line != "":
        tmp = []
        tmp=line.split(" ")
        nb_kw.append(int(tmp[0]))
        lex_discr.append(float(tmp[1]))
        lex_discr_add.append(float(tmp[2]))
        dkm.append(float(tmp[3]))
        line = datafile.readline()

    lines = []
    fig, ax = plt.subplots()
    fig.set_size_inches(4.5, 3.5)
    #ax.figure(num=None, figsize=(3.5, 1.5), dpi=10, facecolor='w', edgecolor='k')
    lines += ax.plot(nb_kw, lex_discr,colors[0], label='Discriminating', linewidth=2)
    lines += ax.plot(nb_kw, lex_discr_add,colors[1], label='Discriminating, Add Class', linewidth=2)
    lines += ax.plot(nb_kw, dkm,colors[4], label='Deep K-Means', linestyle=':', linewidth=1)

    x = []
    for i in range(len(nb_kw)):
        x.append(str(nb_kw[i]))
        x.append('')

    fig.gca().set_xlim(1,5)
    fig.gca().xaxis.set_ticklabels(x)
    
    fig.legend(lines[:2], ['Discriminating', 'Discriminating, Add Class'],
                        loc='upper left', frameon=False, prop={'size': 5.5})
    fig.legend(lines[2:], ['Deep K-Means'],
                        loc='upper right', frameon=False, prop={'size': 5.5})
    plt.xlabel('Number of keywords per classes')
    plt.ylabel('Accuracy')
    fig.savefig(sys.argv[2], bbox_inches='tight')

    plt.close()
