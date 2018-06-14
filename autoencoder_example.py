import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt

startTime = time.time()
mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

learning_rate = 0.01
training_epoch = 20
batch_size = 100

n_hidden = 256
n_input = 28 * 28

X = tf.placeholder(tf.float32, [None, n_input])

W_encode = tf.Variable(tf.random_normal([n_input, n_hidden]))
b_encode = tf.Variable(tf.random_normal([n_hidden]))

encoder = tf.nn.sigmoid(tf.add(tf.matmul(X, W_encode), b_encode))

W_encode = tf.Variable(tf.random_normal([n_hidden, n_input]))
b_encode = tf.Variable(tf.random_normal([n_input]))

decoder = tf.nn.sigmoid(tf.add(tf.matmul(encoder, W_encode), b_encode))

cost = tf.reduce_mean(tf.pow(X - decoder, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

    total_batch = int(mnist.train.num_examples / batch_size)

    for epoch in range(15):
        total_cost = 0
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            _, cost_val = sess.run([optimizer, cost],
                                   feed_dict={X: batch_xs})
            total_cost += cost_val

        print('Eposh:', '%04d' % (epoch + 1),
              'Avg. cost=', '{:.3f}'.format(total_cost/total_batch))
    else:
        print('최적화 완료')

    sample_size = 10
    samples = sess.run(decoder, feed_dict={X: mnist.test.images[:sample_size]})

    fig, ax = plt.subplots(2, sample_size, figsize=(sample_size, 2))

    for i in range(sample_size):
        ax[0][i].set_axis_off()
        ax[1][i].set_axis_off()
        ax[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
        ax[1][i].imshow(np.reshape(samples[i], (28, 28)))

    plt.show()

    labsTime = time.time() - startTime
    print("실행에 소요된 시간=", labsTime, '초')
    if labsTime / 60 > 1: print(labsTime / 60, '분')
