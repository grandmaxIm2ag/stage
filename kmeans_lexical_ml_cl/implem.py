#!/usr/bin/env python
#-*- coding: utf-8 -*-

import numpy as np
import random
import math
import tensorflow as tf

class Autoencoder:

    ###################################################
    # Constructeur de la classse Autoencoder          #
    #                                                 #
    # n            La taille de l'entrée              #
    # n_hidden     La taille de la couche cachée      #
    # n_encode     La taille de la couche d'encodage  #
    # batch        Jeu de données pour l'apprentissage#
    ###################################################
    def __init__(self, n, n_hidden, n_encode, batch):
        self.n = n
        self.n_hidden = n_hidden
        self.n_encode = n_encode
        self.batch = batch

    ###################################################
    # Initialise les placeholders X et Y              #
    ###################################################
    def init_placeholder(self):
        self.X = tf.placeholder(tf.float32, shape=[None,self.n], name = 'X')
        self.Y = tf.placeholder(tf.float32, shape=[None,self.n], name = 'Y')

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
        self.hidden1_layer = tf.nn.relu(tf.add(tf.matmul\
                                               (self.X,\
                                                self.weights['h1']),\
                                                self.biases['b1']))
        self.hidden2_layer = tf.nn.relu(tf.add(tf.matmul\
                                               (self.weights['encode'],\
                                                self.weights['h2']),\
                                                self.biases['b2']))
        self.encode_layer = tf.nn.relu(tf.add(tf.matmul\
                                              (self.weights['h1'],\
                                               self.weights['encode']),\
                                               self.biases['encode']))
        self.decode_layer = tf.nn.relu(tf.add(tf.matmul\
                                              (self.weights['h2'],\
                                               self.weights['decode']),\
                                               self.biases['decode']))


    ###################################################
    # Initialise les fonctions de pertes :            #
    #   * reconstruction                              #
    #   * sparse penalties                            #
    ###################################################
    def init_losses(self):
        self.losses = {
            'rec': tf.reduce_mean(tf.squared_difference(self.decode_layer,\
                                                        self.batch)),
            'lexical': tf.reduce_mean(tf.squared_difference(self.decode_layer,\
                                                            self.batch)),
            'ml': tf.reduce_mean(tf.squared_difference(self.decode_layer,\
                                                       self.batch)),
            'cl': tf.reduce_mean(tf.squared_difference(self.decode_layer,\
                                                       self.batch))
        }

    def train(self, epoch, rate):
        pass
        
if __name__ == "__main__":
    autoencoder = Autoencoder(5,10,100,np.array([[1,2,3,4,5]]))
    autoencoder.init_placeholder()
    autoencoder.init_weights()
    autoencoder.init_biases()
    autoencoder.init_layers()
    autoencoder.init_losses()
