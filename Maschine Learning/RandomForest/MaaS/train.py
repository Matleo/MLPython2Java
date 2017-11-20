import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.ensemble import RandomForestClassifier
from time import time
import random
from sklearn2pmml import PMMLPipeline
from sklearn2pmml import sklearn2pmml
from sklearn.externals import joblib
import json


def load_mnist(test_sample_size):
    custom_data_home = "../../Data/mnist_sklearn"
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


def getPredictions(picCat):
    predictions = []
    for i in range(10):
        path = '../../Data/Own_dat/' + picCat + '-' + str(i) + '.json'

        with open(path) as file:
            pngArray = json.load(file)["pixelValues"]

        prediction = int(clf.predict([pngArray])[0])
        predictions.append(prediction)
    return predictions


def saveStatistics(test_data):
    dic = {}
    dic["accuracy"] = clf.score(test_data["data"], test_data["target"])

    picPredictions = {}
    picCats = ["MNIST", "Computer", "Handwritten", "Font"]
    for picCat in picCats:
        picPredictions[picCat] = getPredictions(picCat)
    dic["picPredictions"] = picPredictions

    with open(export_dir + "statistics_" + str(n_estimators) + ".json", "w") as outfile:
        json.dump(dic, outfile)


if __name__ == "__main__":
    n_estimators = 50
    min_split = 2
    export_dir = "./export/"

    train_data, test_data = load_mnist(10000)
    print("Loaded MNIST data and split into train and test data.")

    # training takes 224s for n=(500,5) -> pmml is 1,88gb big
    # training takes 22s for n=(50,2) -> pmml is 0,26gb big
    clf = RandomForestClassifier(n_estimators=n_estimators, max_features=28, criterion="gini", min_samples_split=min_split)
    mnist_pipeline = PMMLPipeline([
        ("classifier", clf)
    ])
    print("Fitting training data into the RandomForest (this might take a while) ...")
    t0 = time()
    mnist_pipeline.fit(train_data["data"], train_data["target"])
    print("The training took %s seconds to finish.\n" % (round(time() - t0, 2)))

    # convertion takes 808s for n=(500,5)
    # convertion takes 61s for n=(50,2)
    t0 = time()
    sklearn2pmml(mnist_pipeline, export_dir + "RandomForestMNIST_" + str(n_estimators) + ".pmml", with_repr=True)
    print("The convertion took %s seconds to finish.\n" % (round(time() - t0, 2)))

    saveStatistics(test_data)