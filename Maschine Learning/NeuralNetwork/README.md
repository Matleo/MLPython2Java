# Neural Networks Python
     
There are several different technologies to build and train a neural network in Python. I have chosen to focus on the following three:

* Low level [Tensorflow](https://github.com/Matleo/MLPython2Java/tree/develop/Maschine%20Learning/NeuralNetwork/Tensorflow/MNISTClassifier) API
* High level Tensorflow [Estimator](https://github.com/Matleo/MLPython2Java/tree/develop/Maschine%20Learning/NeuralNetwork/Estimator/MNISTClassifier) API
* [Keras](https://github.com/Matleo/MLPython2Java/tree/develop/Maschine%20Learning/NeuralNetwork/Keras) API with Tensorflow backend

For each technology i have trained the following models on the MNIST dataset: 
1. A simple **Feed Forward Neural Network(FFNN)**, which consists of a set of dense layers, where each neuron is connected to each neuron of the following layer
2. A **Convolutional Neural Network(CNN)**, with multiple 2-dimensional convolution and pooling layers, followed by one or more dense layers

In the Tensorflow/Estimator/Keras sub-folders, I will describe in detail how to save a trained model and how to reload it into Python. After saving a model, you can offer a prediction service as a RESTful API, which is described in the [Serving](https://github.com/Matleo/MLPython2Java/tree/develop/Maschine%20Learning/NeuralNetwork/Serving) subfolder.

If you are looking for how to reload a saved model into Java, please refer to [here](https://github.com/Matleo/MLPython2Java/tree/develop/MaschineLearning4J/src/main/java/NeuralNetwork).