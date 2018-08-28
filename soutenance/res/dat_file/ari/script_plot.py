
import matplotlib.pyplot as plt
import sys
import numpy as np


if __name__ == "__main__":
    plt.figure(figsize=(20, 10))
    datafile = open(sys.argv[1], "r")
    line = datafile.readline()
    colors = ['b','g','r','c','m','y','w']
    iColor=0
    sim_lp = []
    sim_sp = []
    mask_lp = []
    nb_kw = []
    dkm = []
    mask_sp = []
    ae = []
    while line != "":
        print(line)
        tmp = []
        tmp=line.split(" ")
        nb_kw.append(int(tmp[0]))
        sim_lp.append(float(tmp[3]))
        sim_sp.append(float(tmp[4]))
        mask_lp.append(float(tmp[1]))
        mask_sp.append(float(tmp[2]))
        dkm.append(float(tmp[5]))
        ae.append(float(tmp[6]))
        line = datafile.readline()

    lines = []
    fig, ax = plt.subplots()
    fig.set_size_inches(4.5, 3.5)
    #ax.figure(num=None, figsize=(3.5, 1.5), dpi=10, facecolor='w', edgecolor='k')
    lines += ax.plot(nb_kw, sim_lp,colors[0], label='SIM LP', linewidth=2)
    lines += ax.plot(nb_kw, sim_sp,colors[1], label='SIM SP', linewidth=2)
    lines += ax.plot(nb_kw, mask_lp,colors[2], label='MASK SP', linewidth=2)
    lines += ax.plot(nb_kw, mask_sp,colors[3], label='MASK SP', linewidth=2)
    #lines += ax.plot(nb_kw, ae,colors[5], label='Autoencoder + K-Means', linewidth=2)
    #lines += ax.plot(nb_kw, dkm,colors[4], label='Deep K-Means', linestyle=':', linewidth=1)
    
    x = []
    for i in range(len(nb_kw)):
        x.append(str(nb_kw[i]))
        x.append('')

    fig.gca().set_xlim(1,5)
    fig.gca().xaxis.set_ticklabels(x)
    #fig.legend(loc='upper left', frameon=False)
    
    fig.legend(lines[:2], ['SIM LP', 'SIM SP'],
                        loc='upper left', frameon=False, prop={'size': 5.5})
    fig.legend(lines[2:4], ['MASK LP', 'MASK SP'],
                        loc='upper center', frameon=False, prop={'size': 5.5})
    #fig.legend(lines[4:], ['Autoencoder + K-Means', 'Deep K-Means'],
    #                    loc='upper right', frameon=False, prop={'size': 5.5})
    plt.xlabel('Number of keywords per classes')
    plt.ylabel('ARI')
    fig.savefig(sys.argv[2], bbox_inches='tight')

    plt.close()
