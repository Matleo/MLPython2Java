import json
import os
import shutil

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from NeuralNetwork.Estimator.MNISTClassifier.FFNN.Wrapper import Wrapper


def getPredictions(Pics):
    predictions = []
    for i in range(0,10):
        path = '../../../../Data/Own_dat/'+Pics+'-'+str(i)+'.png'
        file = tf.read_file(path)
        img = tf.image.decode_png(file, channels=1)
        resized_image = tf.image.resize_images(img, [28, 28])
        tensor=tf.reshape(resized_image, [-1])
        with tf.Session() as sess:
            tArray=1-sess.run(tensor)/255 #von [0,255] auf [0,1] umdrehen
        pred = predictNumberEstimator(tArray)
        predictions.append(pred)
    return predictions

def predictNumberEstimator(tArray):
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"input":np.array([tArray])},
        batch_size=1,
        num_epochs=1,
        shuffle=False)
    predictionObject = classifier.predict(input_fn=predict_input_fn)
    for index,prediction in enumerate(predictionObject):
        guess = prediction["class_ids"][0]
        return int(guess)

def saveConfig(export_dir):
    if os.path.exists(export_dir):
        shutil.rmtree(export_dir)
    inputs = {"input": tf.placeholder(shape=[None,784], dtype=tf.float32, name="input")}
    classifier.export_savedmodel(export_dir,serving_input_receiver_fn=tf.estimator.export.build_raw_serving_input_receiver_fn(inputs))


    #statistics:
    diction = {}
    diction["steps"] = int(steps)
    diction["accuracy"] = round(float(accuracy),4)

    picCategories = ["Handwritten","Computer","MNIST"]
    picDic = {}
    for picCat in picCategories:
        predictions = getPredictions(picCat)
        picDic[picCat] = predictions
    diction["picPredictions"] = picDic

    timestamp=os.listdir(export_dir)[0] #first entry in directory
    accFile = export_dir+"/"+timestamp+"/statistics.json"
    with open(accFile,"w") as outfile:
        json.dump(diction,outfile)
    print("\nSaved Configuration to dir: ./%s" % export_dir)


save = True
#------------------------------------------------------------------------------------

mnist = input_data.read_data_sets("../../../../Data/MNIST_data/")


feature_columns = [tf.feature_column.numeric_column(key="input",dtype=tf.float32, shape=784)]

tempDir="/tmp/mnist_model"
classifier = Wrapper(feature_columns=feature_columns,
                     hidden_units=[550 , 300 , 80],
                     n_classes=10,
                     model_dir=tempDir)

train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"input":np.array(mnist.train.images)},
    y=np.array(mnist.train.labels).astype(np.int32),
    batch_size=100,
    num_epochs=None,
    shuffle=True)

print("Start training...")
steps=1000
classifier.train(input_fn=train_input_fn,steps=steps)
#Epoch is when your model goes through your whole training data once.
# Step is when your model trains on a single batch (or a single sample if you send samples one by one).
# Training for 5 epochs on a 1000 samples 10 samples per batch will take 500 steps

test_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"input":np.array(mnist.test.images)},
    y=np.array(mnist.test.labels).astype(np.int32),
    batch_size=mnist.test.images.size,
    num_epochs=1,
    shuffle=False)
print("Evaluating accuracy...")
accuracy = classifier.evaluate(test_input_fn)["accuracy"]
print("\nTest Accuracy: {0:f}\n".format(accuracy))

if save == True:
    saveConfig("export")
