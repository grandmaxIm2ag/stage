from sklearn.metrics.cluster import normalized_mutual_info_score
import math
import sys
import numpy as np
import numpy.linalg as lin
import subprocess
import random as rand
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

#Initialisation des cluster
def init_centroids(C, k, N):
    centroids = []
    for i in range(0, k):
        #Les representant sont choisis aleatoireent
        centroids.append(C[rand.randint(0, N-1)])
    return centroids

def iMin(tab):
    res = 0
    min = tab[0]
    for i in range (1, len(tab)):
        if (min > tab[i]):
            res = i
            min = tab[res]
    return res

#Renvoie l'indice du cluster auquel est assigne x
def assign(x, centroids,k):
    dist = []
    for i in range(0, k):
        dist.append(euclidian_dist(centroids[i], x))
    return iMin(dist)

#Renvoie la distance euclidienne entre x1 et x2
def euclidian_dist(x1, x2):
    s = 0
    for i in range(0, len(x1)):
        s += pow((x1[i]-x2[i]), 2)
    return math.sqrt(s)


def plot(C, S, t, cent):
    #Projection des donnees dans le plan PCA
    pca = PCA(n_components=2)
    x_r = pca.fit(C).transform(C) #C apres application du PCA
    c_r = pca.fit(C).transform(cent)
    #Visualisation des clusters
    k = len(set(S)) #Nombre de cluster
    target_names = range(k) #Label des clusters obtenus
    c=['blue', 'red', 'green']
    plt.figure()
    for i, target_name in zip (range(k), target_names):
        plt.scatter(x_r[S == i, 0], x_r[S == i, 1], label=target_name, color=c[i])
        plt.legend(loc='best')
        plt.title("Iteration "+str(t))
    c_r1, c_r2 = zip(*c_r)
    plt.scatter(c_r1,c_r2, color='black')
    plt.savefig(str(t)+"test.png")
    

def kmeans(C, M, N, k, e):
    #On initalise le vecteur d'assignement
    S = np.zeros(N)
    t=0
    b = True
    centroids = init_centroids(C, k, N)
    while b:
        plot(C, S, t,centroids)
        #On sauvegarde le vecteur d'assigement
        old_S = np.copy(S)

        #On reassigne les donnes dans les nouveaux clusters
        clust = []
        for i in range(0, k):
            clust.append([])
        for i in range(0, N):
            S[i] = assign(C[i], centroids, k)
            clust[int(S[i])].append(C[i])

        #Calcul des centroids
        for i in range(0, k):
            tmp = np.zeros(M)
            for j in range (0, len(clust[i])):
                tmp += clust[i][j]
            centroids[i] = tmp/len(clust[i])
        
        b = lin.norm(S-old_S) > e
        t+=1    
    return S

if __name__ == "__main__":
    dataset = load_iris()
    C = dataset.data
    M = C[0].shape[0]
    N = C.shape[0]
    k = int(sys.argv[1])
    e = float(sys.argv[2])
    S = kmeans(C, M, N, k, e)
    #Calcule nmi
    y = dataset.target #On recupert les clusters reels
    score = normalized_mutual_info_score(S, y)
    print score
