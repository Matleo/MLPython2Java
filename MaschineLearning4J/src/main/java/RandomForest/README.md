# Using a saved Random Forest in Java
In the following I will describe, how to load a PMML file, which represents a `sklearn.ensemble.RandomForestClassifier` into Java and start making predictions there. If you haven't read about how to perform the export from Python to the PMML file, please read [this](https://github.com/Matleo/MLPython2Java/tree/develop/Maschine%20Learning/RandomForest/MaaS).
## Project setup
### Prerequesites
We will be using the [JPMML-Evaluator](https://github.com/jpmml/jpmml-evaluator), to read the PMML file into Java, and the [Commons CLI](http://commons.apache.org/proper/commons-cli/index.html), to simplify evaluating the program input arguments.

If you are using maven, you can install these libraries by adding the following dependencies to your `pom.xml`:
```maven
<dependency>
	<groupId>org.jpmml</groupId>
	<artifactId>pmml-evaluator</artifactId>
	<version>1.3.9</version>
</dependency>
<dependency>
	<groupId>commons-cli</groupId>
	<artifactId>commons-cli</artifactId>
	<version>1.4</version>
</dependency>
```

### Class structure

* **MNISTClassifier.java**:
The main class for the MNIST classification, to execute loading the whole model from the PMML file, calling for a prediction using the `JPMML Evaluator` and launching a benchmark test.
* **RandomForestWrapper.java**:
This class implements the *Model as a Service* idea. It loads an `Evaluator` from a given PMML file and offers methods to predict the displayed digit of a given png file or compare the Java and Python results, to determin wheter the import was succesfull.
* **BenchmarkTest.java**: This classes only usage is to run a benchmark test on the loaded `Evaluator` to measure it's performance.

## Usage
You can run the `MNISTClassifier.main()` method without passing any program arguments, or pass in a number of optional parameters, according to this `help`:
```java
usage: MNISTClassifier
 -b,--benchmark-test   pass, if you want to run and print the benchmark
                       test
 -c,--compare          pass, if you want the results from Python and Java
                       to be compared for sameness. This action cannot be
                       performed for the R-technology
 -h,--help             print this info message again
 -n,--n-trees <arg>    number of decision trees that were used to create
                       the PMML. Please check for which number of trees,
                       you have a valid PMML file on your system. Default
                       value will be 50
 -p,--picture <arg>    name of the picture, to predict it's displayed
                       digit. <Arg> can either be a full path or a file in
                       the Data/Own_dat directory of this project. Default
                       value will be Data/Own_dat/MNIST-7.png
 -r,--use-R            pass, if the pmml you want to load is from a model,
                       created with R
```

The program will do the following 4 things:
1. Load an `org.jpmml.evaluator.Evaluator` from the PMML file
2. *optional:* Compare the previously saved Python example results with the Java results, calculated by the `Evaluator`
3. Predict the number of the given png file
4. *optional:* Run prediction for 2000 MNSIT datapoints, calculate various statistics and print these to a `benchmark.html` file in this directory.

If you do not pass in any program arguments, the console output should look like following: 
```java
Creating an evaluator from PMML file: ../Maschine Learning/RandomForest/MaaS/export/RandomForestMNIST_50.pmml. 
Depending on the size of the RandomForest, this might take a while...
Finished creating the evaluator! Took 4428ms to finish.

The prediction call for given png, using the Random Forest, took 523ms. (reading the pixel information included)
--> The given picture at "../Maschine Learning/Data/Own_dat/MNIST-7.png" is probably a: 7
```

## How it works
### Loading an Evaluator from PMML
When instantiating a `RandomForestWrapper.java`, you will have to pass a String to the constructor, representing the path of the PMML file from where to load the saved Random Forest. From that file, the `Evaluator` can be loaded as follows: (I have omitted the try-catch blogs for readability)
```java
	InputStream is = new FileInputStream(pmmlFilepath);
	PMML pmml = org.jpmml.model.PMMLUtil.unmarshal(is);
	ModelEvaluatorFactory modelEvaluatorFactory = ModelEvaluatorFactory.newInstance();
	ModelEvaluator<?> modelEvaluator = modelEvaluatorFactory.newModelEvaluator(pmml);
	this.evaluator = modelEvaluator; //class attribute
```

### Making a prediction with the Evaluator
After creating the `Evaluator`, we can now use it to pass input parameters to our previously trained model, and get the output value prediction. As a reminder, [this](https://github.com/Matleo/MLPython2Java/tree/develop/Maschine%20Learning/RandomForest/MaaS) is how we have built our Random Forest model.

#### Preparing inputArguments
Our model wants to be fed with a 784-vector, containing the grayscale pixel values of a picture, displaying a digit. So at first we will have to read the pixel information into a `float[]`. We can accomplish this, reusing the `readPic(String pic)` function from our [TensorflowUtilities](https://github.com/Matleo/MLPython2Java/blob/develop/MaschineLearning4J/src/main/java/NeuralNetwork/Tensorflow/MNIST/TensorflowUtilities.java) class. Now we have to feed this float[] to our previously loaded `Evaluator`:
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
**Note**: `inputFields` contains `InputFields` with names between `x1` and `x784`, but not every value will be present. So `inputFields.length < 784`. This originates from our Random Forest model, trained on the MNIST dataset. Not every pixel of the input png files are relevant, for example the border of every picture will be white, so we do not generate any information of that. In particular, the `feature_importance` of these pixels is 0. That is why those pixels, or features, are not included in the PMML file.

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
In order to know if the entire export and import of the Python model was succesfull, I am comparing the predictions for the saved pictures in the [Data/Own_dat](https://github.com/Matleo/MLPython2Java/tree/develop/Maschine%20Learning/Data/Own_dat) folder. 

One problem I ran into was that reading the grayscale pixel values of a png file in Python and Java will give you different results, which makes it impossible to correctly compare the Java and Python models. Therefore, I used this Python [script](https://github.com/Matleo/MLPython2Java/blob/develop/Maschine%20Learning/Data/Own_dat/saveJsons.py), to read the png files into JSON arrays, so that I could later use these JSON arrays as input for the Python and Java model for comparison. 

As I saved the predictions from the original Python model into a `statistics.json` file, I can now read these predictions, make new predictions in Java, using the `Evaluator` just like above, and finally compare these predictions.

If all of the 40 predictions match, I think it is safe to say that the model transfer was successfull and the Java `Evaluator` represents the original `RandomForestClassifier` properly.