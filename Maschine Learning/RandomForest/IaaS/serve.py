from flask import Flask, request, jsonify
from sklearn.externals import joblib
import cv2  # pip install opencv-python
import numpy as np
from sklearn.datasets import fetch_mldata
import random

app = Flask(__name__)


@app.route("/predict", methods=['POST'])
def predict():
    if not "modelType" in app.config:
        load_model()
    clf = app.config.get("clf")

    json = request.get_json()
    picArray = np.array(json["picArray"]).astype(np.uint8)  # need to change dtype, because cv2 cant handle np.int32
    reshaped_Array = reshapePic(picArray)
    return predictPic(reshaped_Array, clf)


@app.route("/")
def predict_example():
    if not "clf" in app.config:
        load_model()
    clf = app.config.get("clf")

    picName = "MNIST"
    i = 0
    path = '../../Data/Own_dat/' + picName + '-' + str(i) + '.png'
    pngArray = cv2.imread(path, 0)  # int(0,255) where 255 is white
    reshaped_array = reshapePic(pngArray)

    return predictPic(reshaped_array, clf)


def predictPic(pngArray, clf):
    prediction = int(clf.predict([pngArray])[0])
    score = clf.predict_proba([pngArray])[0]
    predProb = round(score[prediction] * 100, 2)

    params = app.config["modelMetaData"]
    return jsonify(_modelMetaData=params, _prediction=int(prediction), _probability=float(predProb))


def reshapePic(pngArray):
    reshaped_Array = cv2.resize(pngArray, (28, 28))
    reshaped_Array = 1 - reshaped_Array / 255
    reshaped_Array = reshaped_Array.flatten()
    return reshaped_Array


def load_model():
    file = 'export.pkl'
    clf = joblib.load(file)
    app.config["clf"] = clf

    allParams = clf.get_params(False)
    params = {}
    params["_modelType"] = clf.__class__.__name__
    params["min_samples_split"] = allParams["min_samples_split"]
    params["n_estimators"] = allParams["n_estimators"]
    params["criterion"] = allParams["criterion"]

    train_data, test_data = load_mnist(10000)
    params["_accuracy"] = clf.score(test_data["data"], test_data["target"])

    app.config["modelMetaData"] = params


def load_mnist(test_sample_size):
    custom_data_home = "/tmp/mnist_sklearn"
    mnist = fetch_mldata("MNIST original", data_home=custom_data_home)  # pixel values as int(0,255) where 0 is white
    mnist.data = mnist.data.astype(float)  # convert to float
    for i in range(len(mnist.data)):
        mnist.data[i] = mnist.data[i] / 255
    return split_data(mnist, test_sample_size)


def split_data(mnist, test_sample_size):
    random.seed(123)
    indices = random.sample(range(len(mnist.data)), test_sample_size)  # random indices in range(mnist.length)
    test_data = {}
    test_data["data"] = mnist.data[indices]
    test_data["target"] = mnist.target[indices]

    train_data = {}
    train_data["data"] = np.delete(mnist.data, indices, axis=0)
    train_data["target"] = np.delete(mnist.target, indices)
    return train_data, test_data


if __name__ == "__main__":
    load_model()
    app.run(debug=True, port=8000)
