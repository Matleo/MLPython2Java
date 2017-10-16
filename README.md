# Operationalization of Python ML models
This project was set up by Matthias Leopold as an intern at Zuehlke Engineering AG Schlieren, to gather the options to go live with a trained Python machine learning model.

## Project structure
The project is split into two sub-projects: 
1. [Machine Learning (Python part)](https://github.com/Matleo/MLPython2Java/tree/develop/Maschine%20Learning)
2. [Machine Learning 4J (Java part)](https://github.com/Matleo/MLPython2Java/tree/develop/MaschineLearning4J)

The sub-projects are themselves ordered by model type. The following models were considered *(links go to the Python part)*:
* [Artificial Neural Networks](https://github.com/Matleo/MLPython2Java/tree/develop/Maschine%20Learning/NeuralNetwork)
* [TODO: Naive Bayes]()
* [TODO: Decision Trees]()
* [TODO: Support Vector Machines]()

For each model type two seperate approaches were evaluated:
1. **Model as a Service**: The whole ML model is supposed to be transfered from Python to Java, to execute predictions directly in Java
2. **Inference as a Service**: The ML model is supposed to be deployed from Python and an inference service is supposed to be made available

## Prerequisites
* Python 3.6.2
* Java 1.8
* Tensorflow 1.3
* [TensorFlow Serving prerequisites](https://www.tensorflow.org/serving/setup) 
* **TODO**