from sklearn.externals import joblib
import random
import numpy as np
from sklearn.datasets import fetch_mldata
import cv2

def load_mnist(test_sample_size):
    custom_data_home = "/tmp/mnist_sklearn"
    mnist =fetch_mldata("MNIST original", data_home=custom_data_home) #pixel values as int(0,255) where 0 is white
    mnist.data = mnist.data.astype(float) #convert to float
    for i in range(len(mnist.data)):
        mnist.data[i] = mnist.data[i]/255
    return split_data(mnist,test_sample_size)

def split_data(mnist, test_sample_size):
    random.seed(123)
    indices = random.sample(range(len(mnist.data)),test_sample_size) # random indices in range(mnist.length)
    test_data = {}
    test_data["data"] = mnist.data[indices]
    test_data["target"] = mnist.target[indices]

    train_data = {}
    train_data["data"] = np.delete(mnist.data,indices, axis=0)
    train_data["target"] = np.delete(mnist.target,indices)

    return train_data,test_data

def printPredictions(pics):
    for i in range(10):
        path = '../../Data/Own_dat/' + pics + '-' + str(i) + '.png'

        pngArray = cv2.imread(path,0)
        pngArray= cv2.resize(pngArray,(28,28))
        pngArray = 1- pngArray/255
        pngArray = pngArray.flatten()

        prediction = int(clf.predict([pngArray])[0])
        score = clf.predict_proba([pngArray])[0]
        predProb = round(score[prediction] *100,2)
        print("%i: The given picture is a %d with probability of: %f%%." % (i, prediction, predProb))

train_data,test_data = load_mnist(10000)

file='export.pkl'
clf = joblib.load(file)
print("Loaded classifier from saved file: %s"% file)


print("Evaluating the accuracy against the test data...")
accuracy = clf.score(test_data["data"],test_data["target"])
print("The accuracy of the RandomForest on the test data is: %f%%"%(accuracy*100))

printPredictions("MNIST")