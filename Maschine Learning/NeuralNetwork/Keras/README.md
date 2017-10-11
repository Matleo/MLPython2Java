# Keras
Keras is a high-level neural networks API, written in Python and capable of running on top of TensorFlow. Keras is very convenient to use and quick to learn. 

I will assume that you are familiar with building and training a Keras model, if that is not the case, please refer to the original [documentation](https://keras.io/). 

## Model as a Service
I evaluated to seperate options to export a Keras model from Python and reload it into Java:
1. Save the native Keras model into a .h5 file and reload it with DL4J
2. Get the Tensorflow `session` from the model and save it as a `SavedModel`
### Native Keras
Saving the native Keras model is as easy as:
```python
    model.save("./export/my_model.h5")
```
and works the same with a Keras `Model` as with a Keras `Sequential`.

Loading the model into Python is just as easy:
```python
    model = keras.models.load_model("./export/my_model.h5")
```
You can find my full example for a `Sequential` [here](https://github.com/Matleo/MLPython2Java/blob/develop/Maschine%20Learning/NeuralNetwork/Keras/MNISTClassifier/Sequential/train_dl4j.py), and for a `Model` [here](https://github.com/Matleo/MLPython2Java/blob/develop/Maschine%20Learning/NeuralNetwork/Keras/MNISTClassifier/Model/cnn_train_dl4j.py)

The interesting part happes [here](https://github.com/Matleo/MLPython2Java/tree/develop/MaschineLearning4J/src/main/java/NeuralNetwork/DL4J) at the Java part. 

It is important to note, that you can not use every Keras element, as the Java import with DL4J only supports a restricted amount of [features](https://deeplearning4j.org/keras-supported-features). This list of features will be available with DL4J v.0.9.2 which is not stable at the moment. I used DL4J v.0.9.1, where there were fewer options available. Because of this reason, and because using an additional (small) framework is often unsafe, i reccomend using the second method, to extract the Tensorflow `session` and export a `SavedModel`.

### Exporting a SavedModel

## Inference as a Service
**TODO**