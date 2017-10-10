import numpy as np
import os

import shutil
import tensorflow as tf
import json
from tensorflow.examples.tutorials.mnist import input_data


def getPredictions(Pics):
    predictions = []
    for i in range(0,10):
        path = '../../../Data/Own_dat/'+Pics+'-'+str(i)+'.png'
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
        num_epochs=1,
        shuffle=False)
    predictionObject = classifier.predict(input_fn=predict_input_fn)
    for index,prediction in enumerate(predictionObject):
        guess = prediction["class"]
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


def model_fn(features, labels, mode):
    input = tf.reshape(features["input"], [-1, 28, 28, 1])
    first_hidden_layer = tf.layers.conv2d(inputs=input, filters=32,kernel_size=(5,5), activation=tf.nn.relu)
    second_hidden_layer = tf.layers.max_pooling2d(inputs=first_hidden_layer, pool_size=(2,2),strides=2)
    third_hidden_layer = tf.layers.conv2d(inputs=second_hidden_layer, filters=64,kernel_size=(5,5), activation=tf.nn.relu)
    fourth_hidden_layer = tf.layers.max_pooling2d(inputs=third_hidden_layer, pool_size=(2,2),strides=2)#shape (?,4,4,64)
    flatten_fourth = tf.reshape(fourth_hidden_layer, [-1, 4*4*64])
    fifth_hidden_layer = tf.layers.dense(flatten_fourth,500, activation=tf.nn.relu)
    sixth_hidden_layer = tf.layers.dropout(fifth_hidden_layer, rate=0.5)
    output_layer = tf.layers.dense(sixth_hidden_layer, 10, activation=tf.nn.softmax)

    if mode == tf.estimator.ModeKeys.PREDICT:#if prediction request
        predictions={"probabilities": tf.identity(output_layer,"output"),
                     "class": tf.identity(tf.argmax(output_layer,axis=1),"class")}
        export_outputs = { "serving_default": tf.estimator.export.PredictOutput(predictions)}
        return tf.estimator.EstimatorSpec(
            mode=mode,
            export_outputs=export_outputs,
            predictions=predictions
            )

    loss = tf.losses.softmax_cross_entropy(labels,output_layer)

    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(tf.argmax(labels,axis=1),tf.argmax(output_layer,axis=1))
    }

    optimizer = tf.train.AdagradOptimizer(learning_rate=0.01)
    train_op = optimizer.minimize(
        loss=loss, global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops)



if __name__ == "__main__":
    save = True
    mnist = input_data.read_data_sets("../../../Data/MNIST_data/",one_hot=True)

    feature_columns = [tf.feature_column.numeric_column(key="input",dtype=tf.float32, shape=784)]

    tempDir="/tmp/own_mnist_model"
    classifier = tf.estimator.Estimator(model_fn=model_fn,model_dir=tempDir)


    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"input":np.array(mnist.train.images).astype(np.float32)},
        y=np.array(mnist.train.labels).astype(np.int32),
        batch_size=100,
        num_epochs=None,
        shuffle=True)

    print("Start training...")
    steps=1000
    classifier.train(input_fn=train_input_fn,steps=steps)

    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"input":np.array(mnist.test.images)},
        y=np.array(mnist.test.labels),
        batch_size=mnist.test.images.size,
        num_epochs=1,
        shuffle=False)

    print("Evaluating accuracy...")
    accuracy = classifier.evaluate(test_input_fn)["accuracy"]
    print("Test Accuracy: {0:f}\n".format(accuracy))


    if save==True:
        saveConfig("export")