import tensorflow as tf


# reads a set of pictures from png file and prints prediction for these images
# Pics is a String, fitting the existing pictures in Data/Own_dat. May be any of "Handwritten"/"MNSIT"/"Computer"/"Font".
# These Images are taken from google pictures
def printPredictions(Pics):
    for i in range(0, 10):
        file = tf.read_file('../../../../../Data/Own_dat/' + Pics + '-' + str(i) + '.png')
        img = tf.image.decode_png(file, channels=1)
        resized_image = tf.image.resize_images(img, [28, 28])  # resize to 28x28 pixel
        tensor = tf.reshape(resized_image, [-1])  # flatten
        tArray = 1 - sess.run(tensor) / 255  # cast from [0,255] to [0,1] where 0->1 and 255->0
        determinNumber(tArray, i)  # call actual prediction function


# actual prediction function, where tArray is a 784 vector
def determinNumber(tArray, i):
    inputArray = sess.run(tf.reshape(tArray, [1, 784]))
    prediction = sess.run(y3, feed_dict={x: inputArray, dKeep: 1})
    predictionIndex = sess.run(tf.argmax(y3, 1), feed_dict={x: inputArray, dKeep: 1})
    predictionIndex = list(predictionIndex)[0]  # get int from returned set
    predictionProb = prediction[0][predictionIndex] * 100
    print("%i: The given image is with %f%% a: %d." % (i, predictionProb, predictionIndex))


if __name__ == "__main__":
    import_dir = "./export/model"
    sess = tf.Session()
    saver = tf.train.import_meta_graph(import_dir + ".meta")
    saver.restore(sess, import_dir)

    graph = tf.get_default_graph()
    y3 = graph.get_tensor_by_name("output:0")
    x = graph.get_tensor_by_name("input:0")
    dKeep = graph.get_tensor_by_name("dropoutRate:0")

    printPredictions("MNIST")
