import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.ensemble import RandomForestClassifier
from time import time
import random
from sklearn2pmml import PMMLPipeline
from sklearn2pmml import sklearn2pmml



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
    train_data, test_data = load_mnist(10000)
    print("Loaded MNIST data and split into train and test data.")

    mnist_pipeline = PMMLPipeline([
        ("classifier", RandomForestClassifier(n_estimators=5, min_samples_split=50))
    ])
    print("Fitting training data into the RandomForest (this might take a while) ...")
    t0 = time()
    mnist_pipeline.fit(train_data["data"], train_data["target"])
    print("The training took %s seconds to finish.\n" % (round(time() - t0, 2)))


    sklearn2pmml(mnist_pipeline, "RandomForestMNIST.pmml", with_repr = True)

