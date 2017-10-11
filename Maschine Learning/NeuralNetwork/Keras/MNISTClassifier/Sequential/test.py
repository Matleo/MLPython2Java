import keras
from tensorflow.examples.tutorials.mnist import input_data

if __name__ == "__main__":
    mnist = input_data.read_data_sets('../../../../Data/MNIST_data', one_hot=True)
    model = keras.models.load_model("./export/my_model.h5")

    loss, accuracity = model.evaluate(mnist.test.images, mnist.test.labels)

    print("\naccuracity on test set: %f%%" % (accuracity * 100))
