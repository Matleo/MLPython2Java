# Sklearn RandomForest -> PMML
In this document I will describe, how to train a Random Forest and parse it to PMML so that you can load that PMML into Java later(done [here](https://github.com/Matleo/MLPython2Java/tree/develop/MaschineLearning4J/src/main/java/RandomForest)). I will once again show an example using the MNIST dataset.

## Prerequisites
The key task of parsing an `sklearn` Random Forest to PMML is handled by the [sklearn2pmml](https://github.com/jpmml/sklearn2pmml) library. You can install the module, using `pip`:
```bash
	pip install --user git+https://github.com/jpmml/sklearn2pmml.git
```

## Build Pipeline
Firstly, you will have to build your `RandomForestClassifier`, and build a `sklearn2pmml.PMMLPipeline` around it:
```python
	clf = RandomForestClassifier(n_estimators=10, min_samples_split=50)
	mnist_pipeline = PMMLPipeline([
		("classifier", clf)
	])
```
Notice that you can extend the `PMMLPipeline`, by passing in a `sklearn_pandas.DataFrameMapper`, a `PCA` operation and one or more `sklearn.feature_selection` operations (see the official [example](https://github.com/jpmml/sklearn2pmml)).

## Parsing PMML
After creating the pipeline you can call the `fit()` function on the pipline, passing in the data:
```python
	mnist_pipeline.fit(train_data["data"], train_data["target"])
```
Now you could start using your classifier `clf` like usual, to make prediction in Python:
```python
	#pngArray = read_array_pixel_values_from_png(...)
	clf.predict(pngArray)
```
But instead of doing this, we now want to export the trained `RandomForestClassifier` to PMML. This is the most important step and can be done by using the `sklearn2pmml.sklearn2pmml` function:
```python
	export_file = "RandomForestMNIST.pmml"
	sklearn2pmml(mnist_pipeline, export_file)
```
This will initiate the process to create the PMML representation of the classifier and store it to the newly created "RandomForestMNIST.pmml" file in the working directory. This might take a while to finish. When trying to convert a very big random forest (happend for n_estimators=500, min_samples_split=2), your JVM might run out of heap memory space. In this case, you will need to use the [jpmml-sklearn](https://github.com/jpmml/jpmml-sklearn) Java command line application directly, and increase the heap space by adding the `-Xmx` argument.

Note that I am saving some more statistics in my [example](https://github.com/Matleo/MLPython2Java/blob/develop/Maschine%20Learning/RandomForest/MaaS/train.py), which include the predictions of some stored pictures in the Data/Own_dat directory. These predictions will later be compared with the Java results, to determine if the import worked correctly.

To read on about how to import the created PMML into Java and start using it to make predictions, please go [here](https://github.com/Matleo/MLPython2Java/tree/develop/MaschineLearning4J/src/main/java/RandomForest).