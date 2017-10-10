import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('../../../Data/MNIST_data', one_hot=True)



def determinNumber(tArray,i):
    output=sess.run(tf.reshape(tArray, [1,784]))
    guessed= sess.run(y_conv, feed_dict={x:output,dKeep: 1})
    guessed = sess.run(tf.nn.softmax(guessed))
    guessedIndex= sess.run(tf.argmax(y_conv,1), feed_dict={x:output,dKeep: 1})
    guessedIndex=list(guessedIndex)[0]#um von set auf int zu kommen
    guessedProb= guessed[0][guessedIndex]*100
    print("%i: Die abgebildete Zahl ist zu %f%% eine: %d." % (i,guessedProb,guessedIndex))

def printPredictions(Pics):
    for i in range(0,10):
        file = tf.read_file('../../Data/Own_dat/'+Pics+'-'+str(i)+'.png')
        img = tf.image.decode_png(file, channels=1)
        resized_image = tf.image.resize_images(img, [28, 28])
        tensor=tf.reshape(resized_image, [-1])
        tArray=1-sess.run(tensor)/255 #von [0,255] auf [0,1] umdrehen
        determinNumber(tArray,i)
def evaluate():
    y_ = graph.get_tensor_by_name("y_:0")
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    with sess.as_default():
        print('test accuracy %g' % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, dKeep: 1.0}))


import_dir = "./export"

sess = tf.Session()

tf.saved_model.loader.load(sess, ["serve"], import_dir)


graph = tf.get_default_graph()
y_conv = graph.get_tensor_by_name("output:0")
x = graph.get_tensor_by_name("input:0")
dKeep = graph.get_tensor_by_name("dropoutRate:0")


printPredictions("Handwritten")

evaluate()