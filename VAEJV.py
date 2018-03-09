# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 14:57:22 2018

@author: Danny
"""

import tensorflow as tf

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np


def next_batch(num, data, labels):
    '''
    Return a total of `num` random samples and labels. 
    '''
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

a = np.loadtxt('nJV.txt')

b = np.reshape(np.transpose(a),(len(a[0]),len(a), ))

par = np.transpose(np.loadtxt('Parin.txt'))

Dim = 50


tf.reset_default_graph()

BATCH_SIZE = 128
LR = 0.0005  

X_in = tf.placeholder(dtype=tf.float32, shape=[None, Dim], name='X')
Y = tf.placeholder(dtype=tf.float32, shape=[None, Dim], name='Y')


dec_in_channels = 1
n_latent = 3




#def lrelu(x, alpha=0.3):
#    return tf.maximum(x, tf.multiply(x, alpha))


def encoder(X_in):
   
    with tf.variable_scope("encoder", reuse=None):

        en0 = tf.layers.dense(X_in, 128, tf.nn.tanh)
        en1 = tf.layers.dense(en0, 64, tf.nn.tanh)
        x = tf.layers.dense(en1, 12, tf.nn.tanh)
        mn = tf.layers.dense(x, units=n_latent)
        sd = 0.5 * tf.layers.dense(x, units=n_latent)
        epsilon = tf.random_normal(tf.stack([tf.shape(x)[0], n_latent]))
        z = mn + tf.multiply(epsilon, tf.exp(sd))

        return z, mn, sd


def decoder(sampled_z):
    with tf.variable_scope("decoder", reuse=None):
        de0 = tf.layers.dense(encoded, 12, tf.nn.tanh)
        de1 = tf.layers.dense(de0, 64, tf.nn.tanh)
        de2 = tf.layers.dense(de1, 128, tf.nn.tanh)
        decoded = tf.layers.dense(de2, Dim, tf.nn.sigmoid)

        return decoded


encoded, mn, sd = encoder(X_in)
decoded = decoder(encoded)


img_loss = 1000*tf.reduce_sum(tf.squared_difference(decoded, X_in), 1)
latent_loss = -0.5 * tf.reduce_sum(1.0 + 2.0 * sd - tf.square(mn) - tf.exp(2.0 * sd), 1)
loss = tf.reduce_mean(img_loss + latent_loss)
train = tf.train.AdamOptimizer(LR).minimize(loss)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(20000):
    b_x, b_y = next_batch(BATCH_SIZE,b,par)
    _, encoded_, mn_,sd_,decoded_, loss_ = sess.run([train, encoded,mn,sd, decoded, loss], {X_in: b_x})

    if step % 100 == 0:     # plotting
       print('train loss: %.4f' % loss_)
       
# visualize in 3D plot for AE
view_data = b
encoded_data = sess.run(encoded, {X_in: view_data})
decoded_data = sess.run(decoded, {X_in: view_data})
fig = plt.figure(1)
plt.scatter(par[:,3],encoded_data[:,1])
fig = plt.figure(2)
plt.scatter(encoded_data[:,0], encoded_data[:,1], s=75,  alpha=.5)
cor= np.corrcoef([par[:,0],par[:,1],par[:,2],par[:,3],par[:,4], encoded_data[:,0],encoded_data[:,1]])
fig = plt.figure(3)
N_TEST_IMG = 8

## original data (first row) for viewing
volt = np.log10 (np.linspace(1,10**1.1,50))
for i in range(N_TEST_IMG):
    rnd = np.random.randint(0,2000)
    plt.subplot(2, N_TEST_IMG//2, i+1)
    plt.plot(volt, view_data[i+rnd,:],label='data')
    plt.plot(volt, decoded_data[i+rnd,:],linestyle='--',label ='VAE')
    plt.legend(loc='upper left')
        
        