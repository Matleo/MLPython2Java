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

## Getting started
I will briefly show you how to get one of the experiments started, assuming you have cloned this repository and installed Java, Maven, Python and Pip already.
1. Set up your Python environment (might be done in a `virtualenv`): 
```bash
	pip install tensorflow keras flask
```
2. Build your Java project with Maven, using the dependencies from this [pom.xml](https://github.com/Matleo/MLPython2Java/blob/develop/MaschineLearning4J/pom.xml)

### Model as a Service
To get started with using a pretrained machine learning model from Python in Java, you can follow this workflow:
1. Decide which model you want to use. For this example i will use the [Feed Forward Neural Network](https://github.com/Matleo/MLPython2Java/tree/develop/Maschine%20Learning/NeuralNetwork/Tensorflow/MNISTClassifier/Feed%20Forward%20NN/SavedModel) trained with Tensorflow.
2. *optional*: Retrain the model. As every model has been initially trained and saved, this step is optional. You can train (and export) the model by simply executing the [train.py](https://github.com/Matleo/MLPython2Java/blob/develop/Maschine%20Learning/NeuralNetwork/Tensorflow/MNISTClassifier/Feed%20Forward%20NN/SavedModel/train.py) script of your model. 
3. Run the [test.py](https://github.com/Matleo/MLPython2Java/blob/develop/Maschine%20Learning/NeuralNetwork/Tensorflow/MNISTClassifier/Feed%20Forward%20NN/SavedModel/test.py) script, which will load the saved model and make prediction against 10 saved .png files in the [data](https://github.com/Matleo/MLPython2Java/tree/develop/Maschine%20Learning/Data/Own_dat) folder. The output should look something like this: 
	```python
	0: The given picture is a 0 with probability of: 100.000000%.
	1: The given picture is a 1 with probability of: 99.997020%.
	2: The given picture is a 2 with probability of: 99.762875%.
	3: The given picture is a 3 with probability of: 100.000000%.
	4: The given picture is a 4 with probability of: 99.993777%.
	5: The given picture is a 5 with probability of: 99.921405%.
	6: The given picture is a 6 with probability of: 84.739697%.
	7: The given picture is a 7 with probability of: 100.000000%.
	8: The given picture is a 8 with probability of: 99.999917%.
	9: The given picture is a 9 with probability of: 99.999940%.
	```
4. Now that everything runs in Python as it should, we can start using the saved model for predictions in Java. For that you can run the execution class [MNISTClassifier]((https://github.com/Matleo/MLPython2Java/blob/develop/MaschineLearning4J/src/main/java/NeuralNetwork/Tensorflow/MNIST/MNISTClassifier.java)). If you don't pass any program arguments, it will load the saved model from the Tensorflow Feed Forward Neural Network, calculate the accuracy of the model with the MNIST dataset, classify a few previously saved .png files from the [data](https://github.com/Matleo/MLPython2Java/tree/develop/Maschine%20Learning/Data/Own_dat) folder and compare the results against the Python results. If everything worked correctly, the tail of the output will look something like:
	```java
    	***Success***
		The calculated accuracy on the MNIST dataset in Java and Python match

		Comparing Java and Python picture predictions...
		***Success***
		The python and java predictions match!
    ```
	For more information on how to use this program, please pass in `-h` as program parameter or refer to the [README](https://github.com/Matleo/MLPython2Java/tree/develop/MaschineLearning4J/src/main/java/NeuralNetwork/Tensorflow) 
## Prerequisites
* Python 3.6.2
* Tensorflow 1.3
* Keras 2.0.8
* Flask 0.12.2
* Java 1.8
* **TODO**