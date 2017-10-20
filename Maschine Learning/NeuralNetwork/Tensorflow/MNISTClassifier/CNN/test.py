import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

def determinNumber(tArray, i):
    inputArray = sess.run(tf.reshape(tArray, [1, 784]))
    score = sess.run(y_conv, feed_dict={x: inputArray, dKeep: 1})[0]
    pred = np.argmax(score)
    predProb = score[pred] * 100
    print("%i: The given picture is a %d with probability of: %f%%." % (i, pred, predProb))


def printPredictions(Pics):
    for i in range(0, 10):
        file = tf.read_file('../../../../Data/Own_dat/' + Pics + '-' + str(i) + '.png')
        img = tf.image.decode_png(file, channels=1)
        resized_image = tf.image.resize_images(img, [28, 28])
        tensor = tf.reshape(resized_image, [-1])
        tArray = 1 - sess.run(tensor) / 255  # von [0,255] auf [0,1] umdrehen
        determinNumber(tArray, i)


if __name__ == "__main__":
    mnist = input_data.read_data_sets('../../../../Data/MNIST_data', one_hot=True)
    import_dir = "./export"

    sess = tf.Session()

    tf.saved_model.loader.load(sess, ["serve"], import_dir)

    graph = tf.get_default_graph()
    y_conv = graph.get_tensor_by_name("output:0")
    x = graph.get_tensor_by_name("input:0")
    dKeep = graph.get_tensor_by_name("dropoutRate:0")

    printPredictions("Handwritten")
