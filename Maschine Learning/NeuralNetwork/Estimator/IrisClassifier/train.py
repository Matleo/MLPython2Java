import os
import shutil
import urllib.request as request

import numpy as np
import tensorflow as tf

from NeuralNetwork.Estimator.IrisClassifier.Wrapper import Wrapper


# Data sets
IRIS_TRAINING = "../../Data/IRIS_data/iris_training.csv"
IRIS_TRAINING_URL = "http://download.tensorflow.org/data/iris_training.csv"

IRIS_TEST = "../../Data/IRIS_data/iris_test.csv"
IRIS_TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"


# If the training and test sets aren't stored locally, download them.
if not os.path.exists(IRIS_TRAINING):
    with request.urlopen(IRIS_TRAINING_URL) as url:
        response = url.read()
        with open(IRIS_TRAINING, "w") as f:
            f.write(response.decode('utf-8'))

if not os.path.exists(IRIS_TEST):
    with request.urlopen(IRIS_TEST_URL) as url:
        response = url.read()
        with open(IRIS_TEST, "w") as f:
            f.write(response.decode('utf-8'))

# Load datasets.
training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename=IRIS_TRAINING,
    target_dtype=np.int,
    features_dtype=np.float32)
test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename=IRIS_TEST,
    target_dtype=np.int,
    features_dtype=np.float32)

#-----------------------------------------------------------------------

def serving_input_receiver_fn():
    inputs = {"input": tf.placeholder(shape=[None,4], dtype=tf.float32, name="input")}
    return tf.estimator.export.ServingInputReceiver(inputs, inputs)


def saveConfig(export_dir):
    if os.path.exists(export_dir):
        shutil.rmtree(export_dir)
    feature_spec = {'x': tf.FixedLenFeature(shape=[4], dtype=tf.float32)}
    serving_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)

    classifier.export_savedmodel(export_dir,serving_input_receiver_fn=serving_input_receiver_fn)
    print("Saved Configuration to dir: ./%s\n"%export_dir)





# Specify that all features have real-value data
feature_columns = [tf.feature_column.numeric_column("input", shape=[4])]
# Build 3 layer DNN with 10, 20, 10 units respectively. n_classes: how many target classes
classifier = Wrapper(feature_columns=feature_columns,
                                        hidden_units=[10, 20, 10],
                                        n_classes=3,
                                        model_dir="/tmp/iris_model")
# Define the training inputs
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"input": np.array(training_set.data)},
    y=np.array(training_set.target),
    num_epochs=None,
    shuffle=True)

# Train model.
classifier.train(input_fn=train_input_fn, steps=1000)

# Define the test inputs
test_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"input": np.array(test_set.data)},
    y=np.array(test_set.target),
    num_epochs=1,
    shuffle=False)

# Evaluate accuracy.
accuracy_score = classifier.evaluate(input_fn=test_input_fn)["accuracy"]

print("\nTest Accuracy: {0:f}\n".format(accuracy_score))



saveConfig("export")

