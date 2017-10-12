# Neural Networks Java
In this part of the project i will describe how to import a model that was previously built, trained and exported in Python. If you have not looked at the [Python part](https://github.com/Matleo/MLPython2Java/tree/develop/Maschine%20Learning/NeuralNetwork) yet, i recommend doing so.

I used two seperate methods to export a Python machine learning model:
1. Keras only: Save the model into a .h5 file
2. Save the model as a Tensorflow `SavedModel`

In the [DL4J](https://github.com/Matleo/MLPython2Java/tree/develop/MaschineLearning4J/src/main/java/NeuralNetwork/DL4J) subdirecory, you will find the code to reload a previously saved Keras model with the [DL4J framework](https://deeplearning4j.org/). 

Whereas in the [Tensorflow](https://github.com/Matleo/MLPython2Java/tree/develop/MaschineLearning4J/src/main/java/NeuralNetwork/Tensorflow) subdirecory, you will find the code using the [Java Tensorflow API](https://www.tensorflow.org/api_docs/java/reference/org/tensorflow/package-summary) to import a `SavedModel`.