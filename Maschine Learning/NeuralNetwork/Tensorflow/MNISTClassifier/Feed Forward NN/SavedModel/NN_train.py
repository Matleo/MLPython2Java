import json
import os
import shutil
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def getPredictions(Pics):
    predictions = []
    for i in range(0, 10):
        path = '../../../../../Data/Own_dat/' + Pics + '-' + str(i) + '.png'
        file = tf.read_file(path)
        img = tf.image.decode_png(file, channels=1)
        resized_image = tf.image.resize_images(img, [28, 28])
        tensor = tf.reshape(resized_image, [-1])
        tArray = 1 - sess.run(tensor) / 255  # von [0,255] auf [0,1] umdrehen
        pred = determinNumber(tArray)
        predictions.append(pred)
    return predictions


def determinNumber(tArray):
    output = sess.run(tf.reshape(tArray, [1, 784]))
    guessedIndex = sess.run(tf.argmax(y3, 1), feed_dict={x: output, dKeep: 1})
    guessedIndex = list(guessedIndex)[0]
    return int(guessedIndex)


def saveConfig():
    export_dir = "./export"
    # if directory exists, remove it
    if os.path.exists(export_dir):
        shutil.rmtree(export_dir)

    # signature of inputs and outputs of the model
    signature = tf.saved_model.signature_def_utils.build_signature_def(
        inputs={'input': tf.saved_model.utils.build_tensor_info(x),
                'dropout': tf.saved_model.utils.build_tensor_info(dKeep)},
        outputs={'output': tf.saved_model.utils.build_tensor_info(y3)},
    )
    signatureMap = {
        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature}

    builder = tf.saved_model.builder.SavedModelBuilder(export_dir)  # initiate builder
    builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING], signature_def_map=signatureMap)
    builder.save()  # execute saving

    # save statistics to compare with java results later:
    diction = {}
    diction["steps"] = int(steps)
    diction["accuracy"] = round(float(acc), 4)

    picCategories = ["Handwritten", "Computer", "MNIST", "Font"]
    picDic = {}
    for picCat in picCategories:
        predictions = getPredictions(picCat)
        picDic[picCat] = predictions
    diction["picPredictions"] = picDic
    with open("./export/statistics.json", "w") as outfile:
        json.dump(diction, outfile)

    print("\nSaved Configuration to dir: %s" % export_dir)


if __name__ == "__main__":
    save = True
    mnist = input_data.read_data_sets("../../../../../Data/MNIST_data/", one_hot=True)
    print()
    steps = 1000

    x = tf.placeholder(tf.float32, [None, 784], name="input")

    y_ = tf.placeholder(tf.float32, [None, 10])

    W1 = tf.Variable(tf.truncated_normal([784, 250], stddev=0.1))
    b1 = tf.Variable(tf.zeros([250]))

    W2 = tf.Variable(tf.truncated_normal([250, 70], stddev=0.1))
    b2 = tf.Variable(tf.zeros([70]))

    W3 = tf.Variable(tf.truncated_normal([70, 10], stddev=0.1))
    b3 = tf.Variable(tf.zeros([10]))

    dKeep = tf.placeholder(tf.float32, name="dropoutRate")
    y1d = tf.nn.relu(tf.matmul(x, W1) + b1)
    y1 = tf.nn.dropout(y1d, dKeep)
    y2d = tf.nn.relu(tf.matmul(y1, W2) + b2)
    y2 = tf.nn.dropout(y2d, dKeep)

    y3 = tf.nn.softmax(tf.matmul(y2, W3) + b3)
    y3 = tf.identity(y3, "output")

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y3, labels=y_)
    cross_entropy = tf.reduce_mean(cross_entropy) * 100

    eta = 0.01
    optimizer = tf.train.GradientDescentOptimizer(eta)
    train_step = optimizer.minimize(cross_entropy)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    correct_prediction = tf.equal(tf.argmax(y3, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # start training
    for i in range(steps):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, dKeep: 0.75})

        if i % 500 == 0:
            print("step " + str(i) + ":", end='')
            print(sess.run([accuracy, cross_entropy],
                           feed_dict={x: mnist.train.images, y_: mnist.train.labels, dKeep: 1.0}))

    acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels, dKeep: 1.0})
    print('test accuracy %g' % acc)

    if save == True:
        saveConfig()
