# Using a saved Random Forest in Java
In the following I will describe, how to load a PMML file, which represents a `sklearn.ensemble.RandomForestClassifier` into Java and start making predictions there. If you haven't read about how to perform the export from Python to the PMML file, please read [this](https://github.com/Matleo/MLPython2Java/tree/develop/Maschine%20Learning/RandomForest/MaaS)
## Project setup
### Prerequesites
We will be using the [JPMML-Evaluator](https://github.com/jpmml/jpmml-evaluator), to read the PMML file into Java. 

If you are using maven, you can install the library by adding the following dependency to your `pom.xml`:
```maven
<dependency>
	<groupId>org.jpmml</groupId>
	<artifactId>pmml-evaluator</artifactId>
	<version>1.3.9</version>
</dependency>
```

### Class structure

* **MNISTClassifier.java**:
The main class for the MNIST classification, to execute loading the whole model from the PMML file and making a prediction using the `JPMML Evaluator`.
* **RandomForestWrapper.java**:
This class implements the *Model as a Service* idea. It loads an `Evaluator` from a given PMML file and offers methods to predict the displayed digit of a given png file or compare the Java and Python results, to determin wheter the import was succesfull.

## Usage
You can either run the `MNISTClassifier.main()` method without passing any program arguments, or pass in the full path to a valid png file, or pass in the name of a png file that is contained in the [Data/Own_Dat](https://github.com/Matleo/MLPython2Java/tree/develop/Maschine%20Learning/Data/Own_dat) folder. 

The program will do the following 3 things:
1. Load an `org.jpmml.evaluator.Evaluator` from the PMML file
2. *optional:* Compare the previously saved Python example results with the Java results, calculated by the `Evaluator`
3. Predict the number of the given png file

If you don't want to compare the Java and Python results and only use the program for prediction, pass in `--noEval` as program argument.

If you do not pass in any program arguments, the console output should look like following: 
```java
Creating an evaluator from PMML file: ../Maschine Learning/RandomForest/MaaS/export/RandomForestMNIST_10.pmml. 
Depending on the size of the RandomForest, this might take a while...
Finished creating the evaluator! Took 3495ms to finish.

Comparing Java and Python picture predictions...
***Success***
The Python and Java predictions match!

The prediction call for given png, using the Random Forest, took 194ms. (including reading the pixel information)
--> The given picture at "../Maschine Learning/Data/Own_dat/MNIST-7.png" is probably a: 7
```

## How it works
### Loading an Evaluator from PMML
When instantiating a `RandomForestWrapper.java`, you have to pass the constructor a String to the PMML file, from where to load the saved model. From that file, the `Evaluator` gets loaded like this (I have omitted the try-catch blogs for readability):
```java
	InputStream is = new FileInputStream(pmmlFile);
	PMML pmml = org.jpmml.model.PMMLUtil.unmarshal(is);
	ModelEvaluatorFactory modelEvaluatorFactory = ModelEvaluatorFactory.newInstance();
	ModelEvaluator<?> modelEvaluator = modelEvaluatorFactory.newModelEvaluator(pmml);
	this.evaluator = modelEvaluator;
```

### Making a prediction with the Evaluator
After creating the `Evaluator`, we can now use it to pass input parameters to our previously trained model, and get the output value prediction. As a reminder, [this](https://github.com/Matleo/MLPython2Java/tree/develop/Maschine%20Learning/RandomForest/MaaS) is how we have built our Random Forest model.

#### Preparing inputArguments
Our model wants to be fed with a 784-vector, containing the grayscale pixel values of a picture, displaying a digit. So at first we will have to read the pixel information into an `float[]`. We can accomplish this, reusing the `readPic(String pic)` function from our [TensorflowUtilities](https://github.com/Matleo/MLPython2Java/blob/develop/MaschineLearning4J/src/main/java/NeuralNetwork/Tensorflow/MNIST/TensorflowUtilities.java) class. Now we have to feed this float[] to our previously loaded `Evaluator`:
```java
	float[] picture = TensorflowUtilities.readPic(picPath);
	List<InputField> inputFields = this.evaluator.getInputFields(); //inputFields are named x1-x784

	Map<FieldName, FieldValue> inputArguments = new LinkedHashMap<>();
	for (int i = 0; i < inputFields.size(); i++) {
		FieldName inputFieldName = inputFields.get(i).getName();//xi with i in int(1,784)
		int index = Integer.valueOf(inputFieldName.toString().substring(1)); //i in int(1,784)
		Object rawValue = picture[index - 1]; //-1 because indices of DataFields are in int(1,784) and array indices are in int(0,783)
		FieldValue inputFieldValue = inputFields.get(i).prepare(rawValue);
		inputArguments.put(inputFieldName, inputFieldValue);
	}
```

#### Evaluate and read out results
After we have prepared the data that we want to feed into our model, we can now actually evaluate the model and read the prediction:
```java
	Map<FieldName, ?> results = this.evaluator.evaluate(inputArguments); #actually using the model
    
    
	List<TargetField> targetFields = this.evaluator.getTargetFields();
	TargetField targetField = targetFields.get(0); //we only have one output value
	FieldName targetFieldName = targetField.getName();
	
    ProbabilityDistribution targetFieldValue = (ProbabilityDistribution) results.get(targetFieldName);
	Object predictionObj = targetFieldValue.getResult();
	int prediction = ((Double) predictionObj).intValue();
```

#### Comparing Java and Python predictions
**TODO**