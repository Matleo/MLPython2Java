# Operationalization of python ML models
This project was set up by Matthias Leopold as an intern at Zuehlke Engineering AG Schlieren, to gather the options to go live with a trained Python machine learning model.

## Project structure
The project is split into two sub-projects: 
1. [Machine Learning (Python part)](https://github.com/Matleo/MLPython2Java/tree/develop/Maschine%20Learning)
2. [Machine Learning 4J (Java part)](https://github.com/Matleo/MLPython2Java/tree/develop/MaschineLearning4J)

The sub-projects are themselves ordered by model type. The following models were considered *(links go to the Python part)*:
* [Artificial Neural Networks](https://github.com/Matleo/MLPython2Java/tree/develop/Maschine%20Learning/NeuralNetwork)
* [Decision Trees]()
* [Support Vector Machines]()
* [Naive Bayes]()
* [Association Rule Learning]()

For each model type two seperate approaches were executed:
1. **Model as a Service**: The whole ML model is supposed to be transfered from Python to Java, to execute predictions directly in Java
2. **Inference as a Service**: The ML model is supposed to be deployed from Python and an inference service is supposed to be made available

## Prerequisites
* Python 3.6.2 or newer
* Java 1.8 or newer
* **TODO**