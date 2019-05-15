# tensor is just an array and flow means manipulations
# here w are going to use Mnist dataset beacuse its in the right format
# an imp work in ML to find a right datset taht suits a perticular model, here mnist data set suits prefectly
# training: 60000 sets, testing: 10000 sets
# mnist contain 28 by 28 hand written digit images
# we pass that to neural nets it will figure out the model
# here features are the pixel values that will contain 0 and 1 ie whitespace or something is there
# this will be a feed forward neural net ie we are going to pass data straight through
# then we will compare the output to the intended output and it will be compared by a cost/lost function
# example of cost func is cross entropy
# the work of optimizer is to make the cost func less by backpropagating and manipulating weights
# example of optimizer i AdamOptimizer, Stocastic GD etc
# backpropagation is going back and changing the weights to get intended output
# each epoch is one cycle of feed forword and backpropation

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('/tmp/data', one_hot=True)
# first parameter asks for the directory of train_data
# one hot helps in multi class function
# for example we hv 10 classes from 0-9
# one hot means only one pixel or element is active and rest are not
# for example if we were supposed to do 0=0, 1=1, 2=2 etc
# but one hot will does like this 0=[1,0,0,0,0,0,0,0,0,0]
# 1=[0,1,0,0,0,0,0,0,0,0]
# 2=[0,0,1,0,0,0,0,0,0,0]
# 3=[0,0,0,1,0,0,0,0,0,0]

n_nodes_h1 = 500
n_nodes_h2 = 500  # may not be identical
n_nodes_h3 = 500
# three hidden layers and 500 nodes in each, they can be different
n_classes = 10  # classes means no. of output layers
n_batches = 100
# this will take 100 features at a time and feed directly to neural net, they will get manipulated and then next batch

x = tf.placeholder('float', [None, 784])  # input data,matrix ie height*width, here height we are giving is None and all
#  our data will be in one line, here we are shaping actually
y = tf.placeholder('float')  # label
# placeholdering variable
# Placeholder simply allocates block of memory for future use.
# Later, we can use feed_dict to feed the data into placeholder. By default,
# placeholder has an unconstrained shape, which allows you to feed tensors of different shapes in a session.
# its not necessary but we can define these so it will be very specific ie 28*28=784 pixels wide
# sec para is shape, if any data that is not of this shape, and will be forced to go inside then tensorflow will raise
#  an error
# placeholders are data that are going to be shoved to the network


def print_shape(obj):
    print(obj.get_shape().as_list())
# To get the shape as a list of ints, do tensor.get_shape().as_list()


def neural_network_model(data):
    # we will pass in raw input data
    # formula is: (input_data*weights + baises)

    hidden_1_layer = {'weight': tf.Variable(tf.random_normal([784, n_nodes_h1])),
                      'baises': tf.Variable(tf.random_normal([n_nodes_h1]))}
    # weight will be a matrix(2D array) of height 784(ie for each pixel)
    # and width 500 (ie 500 random unique values for each pixel)
    # baises are something that are added after the weights
    # we have different baise for each node i.e a 1D array of size 500
    # its imp because in act. func as if all data is zero so weights multiplied will also be 0
    # and no neuron will ever fire
    # random normal means not repeated

    hidden_2_layer = {'weight': tf.Variable(tf.random_normal([n_nodes_h1, n_nodes_h2])),
                      'baises': tf.Variable(tf.random_normal([n_nodes_h2]))}
    # sq. brackets is just for arrays
    hidden_3_layer = {'weight': tf.Variable(tf.random_normal([n_nodes_h2, n_nodes_h3])),
                      'baises': tf.Variable(tf.random_normal([n_nodes_h2]))}

    output_layer = {'weight': tf.Variable(tf.random_normal([n_nodes_h3, n_classes])),
                    'baises': tf.Variable(tf.random_normal([n_classes]))}

    print_shape(data)
    l1 = tf.add(tf.matmul(data, hidden_1_layer['weight']), hidden_1_layer['baises'])
    # matmul is matrix multiplication
    # now we will apply activation func
    print_shape(l1)
    l1 = tf.nn.relu(l1)
    # relu is rectified linear ie our activation/threshold function
    print_shape(l1)
    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weight']), hidden_2_layer['baises'])
    print_shape(l2)
    l2 = tf.nn.relu(l2)
    print_shape(l2)
    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weight']), hidden_3_layer['baises'])
    print_shape(l3)
    l3 = tf.nn.relu(l3)
    print_shape(l3)
    output = tf.matmul(l3, output_layer['weight'])+ output_layer['baises']
    # in output layer there will be no adding

    return output
# output will be a one hot array


def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, y))
    # softmax_cross_entropy_with_logits will calculate the difference bet expected output
    # and obtained one as both are in one hot format
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    # cycles feed forward  +  backprop
    hm_epochs = 10

# training of network starts here
    with tf.Session() as sess:  # context manager, see documentation
        sess.run(tf.initialize_all_variables())
        # its just a session will help it run i dunno what exactly it does

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/n_batches)):
                # here we hv total no. of samples and we are dividing it by batches for no. of cycles
                # _ is just a shorthand for a variable that we just not care about
                epoch_x, epoch_y = mnist.train.next_batch(n_batches)
                # it will make chunks magically
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                # c is cost, in this session we are going to run the optimizer with cost
                # as the session runs it will feed x with x's
                epoch_loss += c
            print('Epoch ', epoch, 'completed out of ', hm_epochs, 'loss:', epoch_loss)
# training ends
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        # it will be a bool
        # argmax will return the index of the max value from the array
        # first argument is the tensor and second argument is axis which has be 1 i.e first column

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        # tf.reduce_mean is equivalent to numpy mean
        # casting to float was imp to find mean

        print("Accuracy: ", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
        # i dunno what actually it does
        # maybe it will match out to input, features to labels


train_neural_network(x)





