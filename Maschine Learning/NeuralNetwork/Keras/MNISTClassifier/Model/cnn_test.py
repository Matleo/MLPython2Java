import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

def printPredictions(Pics):
    for i in range(0, 10):
        path = '../../../../Data/Own_dat/' + Pics + '-' + str(i) + '.png'
        file = tf.read_file(path)
        img = tf.image.decode_png(file, channels=1)
        resized_image = tf.image.resize_images(img, [28, 28])
        tensor = tf.reshape(resized_image, [-1])
        tArray = 1 - sess.run(tensor) / 255  # von [0,255] auf [0,1] umdrehen

        #saveJson(tArray,Pics,i)
        determinNumber(tArray, i)


def determinNumber(tArray, i):
    inputArray = sess.run(tf.reshape(tArray, [1, 784]))
    score = sess.run(y, feed_dict={x: inputArray, learningPhase: False})[0]
    predictedIndex = np.argmax(score)
    predictedProb = score[predictedIndex] * 100
    print("%i: Die abgebildete Zahl ist zu %f%% eine: %d." % (i, predictedProb, predictedIndex))

if __name__=="__main__":
    mnist = input_data.read_data_sets('../../../../Data/MNIST_data', one_hot=True)

    import_dir = "./export"
    sess = tf.Session()
    tf.saved_model.loader.load(sess, ["serve"], import_dir)

    graph = tf.get_default_graph()
    y = graph.get_tensor_by_name("output/Softmax:0")
    x = graph.get_tensor_by_name("input_input:0")
    learningPhase = graph.get_tensor_by_name("dropout_1/keras_learning_phase:0")

    printPredictions("MNIST")



