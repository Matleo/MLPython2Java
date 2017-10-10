import keras
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np




mnist = input_data.read_data_sets('../../../../Data/MNIST_data', one_hot=True)
images_test= mnist.test.images.reshape(mnist.test.images.shape[0], 28, 28, 1)


model = keras.models.load_model("./export/my_model.h5")

loss,accuracity=model.evaluate(images_test,mnist.test.labels,batch_size=len(mnist.test.images))

print("accuracity on test set: %f %%"%(accuracity*100))


