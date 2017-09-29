import tensorflow as tf
import numpy
import os

import_super_dir = "./export/"
timestamp=os.listdir(import_super_dir)[0] #first entry in directory
import_dir = import_super_dir+timestamp

predict_fn = tf.contrib.predictor.from_saved_model(import_dir)


# Classify two new flower samples.
new_samples = numpy.array(
    [[6.4, 3.2, 4.5, 1.5],
     [6.5,3.0,5.2,2.0]],
    dtype=numpy.float32)

predictions = predict_fn({"input": new_samples})

predicted_classIDs = [numpy.argmax(p) for p in predictions["scores"]]
predicted_classes = [None]*len(predicted_classIDs)
for i in range(0,len(predicted_classIDs)):
    if predicted_classIDs[i] == 0:
        predicted_classes[i] = "Iris setosa"
    if predicted_classIDs[i] == 1:
        predicted_classes[i] = "Iris versicolor"
    if predicted_classIDs[i] == 2:
        predicted_classes[i] = "Iris virginica"

print("Class Predictions of new Samples:    {}\n".format(predicted_classes))

