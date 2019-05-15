# convolution neural network
# used in image analysis
# we create a 'feature map' of the given input
# we break an image in pixels and in one convolution step we take 3*3 grid and shift 2 steps
# then we pool. conv+pool=HL

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)


n_classes = 10
batch_size = 128

x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

def print_shape(obj):
    print(obj.get_shape().as_list())


def conv2d(x, W):
    return tf.nn.conv2d(x, W, [1, 1, 1, 1], padding='SAME')
    # this func is just converts into 2 dimentional conv net. i dunno what that means
    # in theory we shifted our grid this work is done by that func
    # strides is we shift one pixel at time
    # for padding see video its easy


def maxpool2d(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # pooling by 2 X 2 pixel at a time
    # ksize is actual size of the window and strides is the movement of window


def convolutional_neural_network(x):
    weights = {'W_Conv1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
               # its gonna take 5 X 5 convolution, take one input, and gonna produce 32 outputs
               'W_Conv2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
               'W_fc': tf.Variable(tf.random_normal([7*7*64, 1024])),
               # we started by 28*28 but using conv we significantly compress that image into feature maps ie 7*7
               # its 7 by 7 times 64 features and 1024 nodes
               'output': tf.Variable(tf.random_normal([1024, n_classes]))}
    # naming of variables are according to Mnist for experts in tenserflow.com

    baises = {'b_Conv1': tf.Variable(tf.random_normal([32])),
              'b_Conv2': tf.Variable(tf.random_normal([64])),
              'b_fc': tf.Variable(tf.random_normal([1024])),
              'output': tf.Variable(tf.random_normal([n_classes]))}
    # baises are just for the no. of output

    x = tf.reshape(x, shape=[-1, 28, 28, 1])  # x is input/orig and reshaping a 784 pixels image to a
                                              # to flat one 28*28 image
    conv1 = tf.nn.relu(conv2d(x, weights['W_Conv1']) + baises['b_Conv1'])
    conv1 = maxpool2d(conv1)
    conv2 = tf.nn.relu(conv2d(conv1, weights['W_Conv2']) + baises['b_Conv2'])
    conv2 = maxpool2d(conv2)

    fc = tf.reshape(conv2, [-1, 7*7*64])
    # we reshape conv2 and we rshape it to & by 7 times 64
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc']) + baises['b_fc'])

    output = tf.matmul(fc, weights['output'] + baises['output'])
    return output


def train_neural_network(x):
    prediction = convolutional_neural_network(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits
                          (logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    hm_epochs = 10

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples / batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x,
                                                              y: epoch_y})
                epoch_loss += c
            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:',
                  epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x: mnist.test.images, y:
                                          mnist.test.labels}))


train_neural_network(x)