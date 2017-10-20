import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#saves the model with tensorflow saver to export dir
def saveConfig():
    export_dir = "./export/model"
    saver = tf.train.Saver()
    saver.save(sess, export_dir)


if __name__ == "__main__":
    save = True
    mnist = input_data.read_data_sets("../../../../../Data/MNIST_data/", one_hot=True)

    # input images: beliebig viele(None) Bilder, wobei jedes auf 784 Pixel abgebildet wurde. Jeder Pixel ist in (0,1), wobei 1 dunkel ist
    x = tf.placeholder(tf.float32, [None, 784], name="input")#name of input tensor: input:0
    #define dropoutRate
    dKeep = tf.placeholder(tf.float32, name="dropoutRate")#name of dropout tensor: dropoutRate:0

    #input labels as onehot vectors
    y_ = tf.placeholder(tf.float32, [None, 10])

    # define weights and biases
    W1 = tf.Variable(tf.truncated_normal([784, 250], stddev=0.1))
    b1 = tf.Variable(tf.zeros([250]))

    W2 = tf.Variable(tf.truncated_normal([250, 70], stddev=0.1))
    b2 = tf.Variable(tf.zeros([70]))

    W3 = tf.Variable(tf.truncated_normal([70, 10], stddev=0.1))
    b3 = tf.Variable(tf.zeros([10]))


    #define Layers as dense layers with relu activation function
    y1d = tf.nn.relu(tf.matmul(x, W1) + b1)
    y1 = tf.nn.dropout(y1d, dKeep)
    y2d = tf.nn.relu(tf.matmul(y1, W2) + b2)
    y2 = tf.nn.dropout(y2d, dKeep)

    #output layer
    y3 = tf.nn.softmax(tf.matmul(y2, W3) + b3)
    y3 = tf.identity(y3, "output") #name of output tensor: output:0


    # loss function: measures how inefficient the predictions are
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y3, labels=y_)
    cross_entropy = tf.reduce_mean(cross_entropy) * 100

    # define optimizer operation
    eta = 0.01
    optimizer = tf.train.GradientDescentOptimizer(eta)
    train_step = optimizer.minimize(cross_entropy)

    sess = tf.InteractiveSession()  # launch the model
    tf.global_variables_initializer().run()  # initialize the variables

    # Vector aus boolean, welcher speichert ob prediction für Bild i korrekt war
    correct_prediction = tf.equal(tf.argmax(y3, 1),
                                  tf.argmax(y_, 1))  # argmax gives index of highest entry along specified axis
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # start training, do steps*batch iterations
    steps = 1000
    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100) #get a batch of 100 next image/label pairs
        # feed_dict replaces the placeholder tensors x and y_ with the training examples
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, dKeep: 0.75})

        # print current accuracy against train set and loss
        if i % 200 == 0:
            print(sess.run([accuracy, cross_entropy],
                           feed_dict={x: mnist.train.images, y_: mnist.train.labels, dKeep: 1.0}))

    #print final accuracy against test set
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels, dKeep: 1.0}))

    #maybe save the model
    if save == True:
        saveConfig()
