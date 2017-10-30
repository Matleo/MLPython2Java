from flask import Flask, request, jsonify
from sklearn.externals import joblib
import cv2  # pip install opencv-python
import numpy as np

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
    path = '../Data/Own_dat/' + picName + '-' + str(i) + '.png'
    pngArray = cv2.imread(path, 0)  # int(0,255) where 255 is white
    reshaped_array = reshapePic(pngArray)

    return predictPic(reshaped_array, clf)


def predictPic(pngArray, clf):
    prediction = int(clf.predict([pngArray])[0])
    score = clf.predict_proba([pngArray])[0]
    predProb = round(score[prediction] * 100, 2)

    return jsonify(prediction=int(prediction), probability=float(predProb))


def reshapePic(pngArray):
    reshaped_Array = cv2.resize(pngArray, (28, 28))
    reshaped_Array = 1 - reshaped_Array / 255
    reshaped_Array = reshaped_Array.flatten()
    return reshaped_Array


def load_model():
    file = 'export.pkl'
    clf = joblib.load(file)

    app.config["clf"] = clf


if __name__ == "__main__":
    load_model()
    app.run(debug=True, port=8000)
