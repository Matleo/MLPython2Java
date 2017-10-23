from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import os

app = Flask(__name__)
tf.app.flags.DEFINE_string('model', "t_ffnn", 'SavedModel to load for prediction')


@app.route("/predict", methods=['POST'])
def predict():
    if not "modelType" in app.config:
        load_model("t_ffnn")

    req = request.get_json()
    picArray = req["picArray"]
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

    return jsonify(prediction=int(pred), probability=float(predProb))


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


def reshapePic(pic):
    for i in range(len(pic)):
        for j in range(len(pic[i])):
            pic[i][j] = [pic[i][j]]
    resized_image = tf.image.resize_images(pic, [28, 28])
    tensor = tf.reshape(resized_image, [-1])

    sess = tf.Session()
    tArray = 1 - sess.run(tensor) / 255  # von [0,255] auf [0,1] umdrehen
    reshaped_Array = sess.run(tf.reshape(tArray, [1, 784]))
    return reshaped_Array

def load_model(model):
    sess = tf.Session()
    import_dir = getImportDir(model)
    tf.saved_model.loader.load(sess, ["serve"], import_dir)
    graph = tf.get_default_graph()

    app.config["modelType"] = model
    app.config["session"] = sess
    app.config["graph"] = graph

if __name__ == "__main__":

    model = tf.app.flags.FLAGS.model
    load_model(model)

    app.run(debug=True, port=8000)