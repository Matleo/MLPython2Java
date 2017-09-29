import tensorflow as tf
import numpy as np
import os
import shutil

from tensorflow.examples.tutorials.mnist import input_data
from NeuralNetwork.Estimator.IrisClassifier.Wrapper import Wrapper

def serving_input_receiver_fn_wrapper():
    inputs = {"input": tf.placeholder(shape=[None,784], dtype=tf.float32, name="input")}
    return tf.estimator.export.ServingInputReceiver(inputs, inputs)
def saveConfig(export_dir):
    if os.path.exists(export_dir):
        shutil.rmtree(export_dir)
    classifier.export_savedmodel(export_dir,serving_input_receiver_fn=serving_input_receiver_fn_wrapper)
    print("Saved Configuration to dir: ./%s\n" % export_dir)


def printPredictions(pics):
    for i in range(0,10):
        path = '../../Data/Own_dat/'+pics+'-'+str(i)+'.png'
        file = tf.read_file(path)
        img = tf.image.decode_png(file, channels=1)
        resized_image = tf.image.resize_images(img, [28, 28])
        tensor=tf.reshape(resized_image, [-1])
        with tf.Session() as sess:
            tArray=1-sess.run(tensor)/255 #von [0,255] auf [0,1] umdrehen
        predictNumberEstimator(tArray,i)
def predictNumberEstimator(tArray, i):
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"input":np.array([tArray])},
        batch_size=1,
        num_epochs=1,
        shuffle=False)
    predictionObject = classifier.predict(input_fn=predict_input_fn)
    for index,prediction in enumerate(predictionObject):
        guess = prediction["class_ids"][0]
        prob = prediction["probabilities"][guess] * 100
        print("%i: Die abgebildete Zahl ist zu %f%% eine: %d." % (i,prob,guess))

save = False
#------------------------------------------------------------------------------------

mnist = input_data.read_data_sets("../../Data/MNIST_data/")

tempDir="/tmp/mnist_model"

feature_columns = [tf.feature_column.numeric_column(key="input",dtype=tf.float32, shape=784)]
classifier = Wrapper(feature_columns=feature_columns,
                     hidden_units=[550,300,80],
                     n_classes=10,
                     model_dir=tempDir)

train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"input":np.array(mnist.train.images)},
    y=np.array(mnist.train.labels).astype(np.int32),
    batch_size=100,
    num_epochs=None,
    shuffle=True)

classifier.train(input_fn=train_input_fn,steps=1000)

test_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"input":np.array(mnist.test.images)},
    y=np.array(mnist.test.labels).astype(np.int32),
    batch_size=mnist.test.images.size,
    num_epochs=1,
    shuffle=False)

accuracy = classifier.evaluate(test_input_fn)["accuracy"]
print("\nTest Accuracy: {0:f}\n".format(accuracy))

if save == True:
    saveConfig("export")
