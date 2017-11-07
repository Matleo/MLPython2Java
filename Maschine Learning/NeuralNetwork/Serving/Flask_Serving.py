from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import os
import json

app = Flask(__name__)
tf.app.flags.DEFINE_string('model', "t_ffnn", 'SavedModel to load for prediction')


@app.route("/predict", methods=['POST'])
def predict():
    if not "modelType" in app.config:
        load_model("t_ffnn")

    json = request.get_json()
    picArray = json["picArray"]
    reshaped_Array = reshapePic(picArray)

    model = app.config.get("modelType")
    graph = app.config.get("graph")
    sess = app.config.get("session")

    return predictPic(reshaped_Array, model, graph, sess)


@app.route("/")
def predict_example():
    if not "modelType" in app.config:
        load_model("t_ffnn")

    model = app.config.get("modelType")
    graph = app.config.get("graph")
    sess = app.config.get("session")

    picName = "MNIST"
    i = 0
    path = '../../Data/Own_dat/' + picName + '-' + str(i) + '.png'
    file = tf.read_file(path)
    pic = tf.image.decode_png(file, channels=1)
    with tf.Session() as session:
        reshaped_Array = reshapePic(session.run(pic))
    return predictPic(reshaped_Array, model, graph, sess)


def predictPic(picArray, model, graph, sess):
    score = []

    if model == "t_ffnn" or model == "t_cnn":
        y = graph.get_tensor_by_name("output:0")
        x = graph.get_tensor_by_name("input:0")
        dKeep = graph.get_tensor_by_name("dropoutRate:0")
        score = sess.run(y, feed_dict={x: picArray, dKeep: 1})[0]
    elif model == "e_ffnn" or model == "e_cnn":
        y = graph.get_tensor_by_name("output:0")
        x = graph.get_tensor_by_name("input:0")
        score = sess.run(y, feed_dict={x: picArray})[0]
    elif model == "k_ffnn" or model == "k_cnn":
        y = graph.get_tensor_by_name("output/Softmax:0")
        x = graph.get_tensor_by_name("input_input:0")
        learningPhase = graph.get_tensor_by_name("dropout_1/keras_learning_phase:0")
        score = sess.run(y, feed_dict={x: picArray, learningPhase: False})[0]

    pred = np.argmax(score)
    predProb = score[pred] * 100
    params = app.config["modelMetaData"]

    return jsonify(_modelMetaData=params, _prediction=int(pred), _probability=float(predProb))


def getImportDir(model):
    if model == "t_ffnn":
        return "../Tensorflow/MNISTClassifier/Feed Forward NN/SavedModel/export"
    if model == "t_cnn":
        return "../Tensorflow/MNISTClassifier/CNN/export"
    if model == "e_ffnn":
        import_super_dir = "../Estimator/MNISTClassifier/FFNN/export/"
        timestamp = os.listdir(import_super_dir)[0]  # first entry in directory
        import_dir = import_super_dir + timestamp
        return import_dir
    if model == "e_cnn":
        import_super_dir = "../Estimator/MNISTClassifier/CNN/export/"
        timestamp = os.listdir(import_super_dir)[0]  # first entry in directory
        import_dir = import_super_dir + timestamp
        return import_dir
    if model == "k_ffnn":
        return "../Keras/MNISTClassifier/Sequential/export"
    if model == "k_cnn":
        return "../Keras/MNISTClassifier/Model/export"

def getModelFullname(model):
    if model == "t_ffnn":
        return "Tensorflow Feed Forward Neural Network"
    if model == "t_cnn":
        return "Tensorflow Convolutional Neural Network"
    if model == "e_ffnn":
        return "Estimator Feed Forward Neural Network"
    if model == "e_cnn":
        return "Estimator Convolutional Neural Network"
    if model == "k_ffnn":
        return "Keras Feed Forward Neural Network"
    if model == "k_cnn":
        return "Keras Convolutional Neural Network"
    print("ERROR: You passed in a wrong model name, it needs to be in {t_ffnn, t_cnn, e_ffnn, e_cnn, k_ffnn, k_cnn}")

def reshapePic(pic):
    # 1. As Tensorflow expects the input Tensor to be channel last, we need to wrap every pixels value into an array individually
    for i in range(len(pic)):
        for j in range(len(pic[i])):
            pic[i][j] = [pic[i][j]]

    resized_image = tf.image.resize_images(pic, [28, 28])  # 2. resize to 28x28
    tensor = tf.reshape(resized_image, [-1])  # 3. flatten

    with tf.Session() as sess:
        tArray = 1 - sess.run(tensor) / 255  # 4. reverse [0,255] to [0,1]
        reshaped_Array = sess.run(tf.reshape(tArray, [1, 784]))  # make batch of size 1
    return reshaped_Array


def load_model(model):
    sess = tf.Session()
    import_dir = getImportDir(model)
    tf.saved_model.loader.load(sess, ["serve"], import_dir)
    graph = tf.get_default_graph()
    app.config["modelType"] = model
    app.config["session"] = sess
    app.config["graph"] = graph

    with open(import_dir+"/statistics.json") as file:
        data = json.load(file)
    params = {}
    params["_modelType"] = getModelFullname(model)
    if model in ["t_ffnn","t_cnn","e_ffnn","e_cnn"]:
        params["steps"] = data["steps"]
        params["batch_size"] = data["batch_size"]
    else:
        params["epochs"] = data["epochs"]
    params["_accuracy"] = data["accuracy"]
    app.config["modelMetaData"] = params


if __name__ == "__main__":
    model = tf.app.flags.FLAGS.model
    load_model(model)

    app.run(debug=True, port=8000)
