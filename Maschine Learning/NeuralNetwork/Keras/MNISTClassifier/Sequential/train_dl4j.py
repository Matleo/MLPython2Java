from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
import keras
import sys
from tensorflow.examples.tutorials.mnist import input_data

if __name__ == "__main__":
    if keras.__version__ != "1.2.2":
        print("You are using a wrong version of keras. Only 1.2.2 is supported!")
        print("Your version: %s" % keras.__version__)
        sys.exit(1)

    mnist = input_data.read_data_sets('../../../../Data/MNIST_data', one_hot=True)

    save = True

    # Building the model
    model = Sequential()
    model.add(Dense(550, activation="relu", input_dim=784))
    model.add(Dropout(0.75))
    model.add(Dense(300, activation="relu"))
    model.add(Dropout(0.75))
    model.add(Dense(80, activation="relu"))
    model.add(Dense(10, activation="softmax"))

    model.compile(optimizer="rmsprop",
                  loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])

    model.fit(mnist.train.images, mnist.train.labels, nb_epoch=3, batch_size=200)

    loss, accuracity = model.evaluate(mnist.test.images, mnist.test.labels, batch_size=len(mnist.test.images))

    print("accuracity on test set: %f %%" % (accuracity * 100))

    if save == True:
        model.save("./export/my_model.h5")
