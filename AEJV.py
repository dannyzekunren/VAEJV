# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 11:56:49 2018

@author: danny
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

# Hyper Parameters
BATCH_SIZE = 256
LR = 0.001         # learning rate



# tf placeholder
tf_x = tf.placeholder(tf.float32, [None, Dim])    # value in the range of (0, 1)

# encoder
en0 = tf.layers.dense(tf_x, 128, tf.nn.tanh)
en1 = tf.layers.dense(en0, 64, tf.nn.tanh)
en2 = tf.layers.dense(en1, 12, tf.nn.tanh)
encoded = tf.layers.dense(en2, 2)

# decoder
de0 = tf.layers.dense(encoded, 12, tf.nn.tanh)
de1 = tf.layers.dense(de0, 64, tf.nn.tanh)
de2 = tf.layers.dense(de1, 128, tf.nn.tanh)
decoded = tf.layers.dense(de2, Dim, tf.nn.sigmoid)

loss = tf.losses.mean_squared_error(labels=tf_x, predictions=decoded)
train = tf.train.AdamOptimizer(LR).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())




for step in range(8000):
    b_x, b_y = next_batch(BATCH_SIZE,b,par)
    _, encoded_, decoded_, loss_ = sess.run([train, encoded, decoded, loss], {tf_x: b_x})

    if step % 100 == 0:     # plotting
        print('train loss: %.4f' % loss_)
        # plotting decoded image (second row)
#       
#        for i in range(N_TEST_IMG):
#            a[1][i].clear()
#            a[1][i].imshow(np.reshape(decoded_data[i], (28, 28)), cmap='gray')
#            a[1][i].set_xticks(()); a[1][i].set_yticks(())
#        plt.draw(); plt.pause(0.01)
#plt.ioff()


# visualize in 3D plot for AE
view_data = b
encoded_data = sess.run(encoded, {tf_x: view_data})
decoded_data = sess.run(decoded, {tf_x: view_data})
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
    plt.plot(volt, decoded_data[i+rnd,:],linestyle='--',label ='AE')
    plt.legend(loc='upper left')
    
#ax = Axes3D(fig)
#X, Y, Z = encoded_data[:, 0], encoded_data[:, 1], encoded_data[:, 2]
#for x, y, z, s in zip(X, Y, Z, test_y):
#    c = cm.rainbow(int(255*s/9))
#    ax.text(x, y, z, s, backgroundcolor=c)
#    ax.set_xlim(X.min(), X.max()); ax.set_ylim(Y.min(), Y.max()); ax.set_zlim(Z.min(), Z.max())
#plt.show()

