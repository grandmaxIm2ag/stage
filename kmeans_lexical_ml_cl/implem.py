#!/usr/bin/env python
#-*- coding: utf-8 -*-

import numpy as np
import random
import math
import tensorflow as tf

###################################################
# Masque le document X avec les mot clés de KW    #
#                                                 #
# X            Le document                        #
# KW           Les indices des mots clés          #
###################################################
def mask(X, KW):
    X_prime = np.zeros(X.shape[0])
    for i in range(KW.shape[0]):
        X_prime[KW[i]]=X[KW[i]]
    return X_prime
        
class Autoencoder:
    
    ###################################################
    # Constructeur de la classse Autoencoder          #
    #                                                 #
    # n            La taille de l'entrée              #
    # n_hidden     La taille de la couche cachée      #
    # n_encode     La taille de la couche d'encodage  #
    # batch        Jeu de données pour l'apprentissage#
    ###################################################
    def __init__(self, n, n_hidden, n_encode, batch, KW):
        self.n = n
        self.n_hidden = n_hidden
        self.n_encode = n_encode
        self.batch = batch
        self.batch_prime = []
        for i in range(batch.shape[0]):
            self.batch_prime.append(mask(batch[i], KW))

    ###################################################
    # Initialise les placeholders X et Y              #
    ###################################################
    def init_placeholder(self):
        self.X = tf.placeholder(tf.float32, shape=[None,self.n], name = 'X')

    ###################################################
    # Initialise les matrices de poids dans le        #
    # dictionnaire weights                            #
    ###################################################
    def init_weights(self):
        with tf.name_scope('weights'):
            self.weights = {
                'h1': tf.Variable(tf.truncated_normal([self.n, self.n_hidden])\
                                  , name="h1"),
                'h2': tf.Variable(tf.truncated_normal\
                                  ([self.n_encode, self.n_hidden]),name="h2"),
                'encode': tf.Variable(tf.truncated_normal\
                                      ([self.n_hidden, self.n_encode]),\
                                      name="encode"),
                'decode': tf.Variable(tf.truncated_normal([self.n_hidden, self.n])\
                                      ,name="decode")
            }

    ###################################################
    # Initialise les vecteurs de biais dans le        #
    # dictionnaire biases                             #
    ###################################################
    def init_biases(self):
        with tf.name_scope('biases'):
            self.biases = {
                'b1': tf.Variable(tf.zeros([self.n_hidden]), name="b1"),
                'encode': tf.Variable(tf.zeros([self.n_encode]), \
                                      name="b_encode"),
                'b2': tf.Variable(tf.zeros([self.n_hidden]), name="b2"),
                'decode': tf.Variable(tf.zeros([self.n]), name="b_decode")
            }


    ###################################################
    # Initialise les couches du réseaux               #       
    ###################################################
    def init_layers(self):
        self.hidden1_layer = tf.nn.sigmoid(tf.add(tf.matmul\
                                               (self.X,\
                                                self.weights['h1']),\
                                                self.biases['b1']))
        self.encode_layer = tf.nn.sigmoid(tf.add(tf.matmul\
                                              (self.hidden1_layer,\
                                               self.weights['encode']),\
                                               self.biases['encode']))
        self.hidden2_layer = tf.nn.sigmoid(tf.add(tf.matmul\
                                               (self.encode_layer,\
                                                self.weights['h2']),\
                                                self.biases['b2']))
        self.decode_layer = tf.nn.softplus(tf.add(tf.matmul\
                                              (self.hidden2_layer,\
                                               self.weights['decode']),\
                                               self.biases['decode']))


    ###################################################
    # Initialise les fonctions de coûts :             #
    #   - reconstruction                              #
    #   - sparse penalties*                           #
    ###################################################
    def init_losses(self):
        self.losses = {
            'rec': tf.reduce_sum(tf.pow(self.X - self.decode_layer, 2))
        }

    ###################################################
    # Fonction d'apprentissage de l'autoencoder       #
    #                                                 #
    # epoches      Nombre d'epoches                   #
    # rate         Pas d'apprentissage                #
    ###################################################
    def train(self, epoches, rate):
        train_step = tf.train.GradientDescentOptimizer(rate).minimize(\
            self.losses['rec'])
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            for e in range(1, epoches):
                sess.run(train_step, feed_dict={self.X: self.batch})
                if e % 100 == 0:
                    print "epoch : "+str(e)
                    pred = sess.run(self.decode_layer,feed_dict=\
                                    {self.X: self.batch})
                    print "Estimation : \n"+str(pred)
                    print '\n'
                    
if __name__ == "__main__":
    batch = np.array([
        [11,4,3,4,5],
        [1,0,3,7,3],
        [8,5,13,9,7]
    ],dtype=np.dtype('Float32'))
    KW = np.array([0, 2])
    autoencoder = Autoencoder(5,10,100,batch, KW)
    autoencoder.init_placeholder()
    autoencoder.init_weights()
    autoencoder.init_biases()
    autoencoder.init_layers()
    autoencoder.init_losses()
    autoencoder.train(40000, 0.01)
