# Serving a Random Forest model with flask

This document will walk you through on how to train, use, and serve a Random Forest machine learning model with `sklearn` and `flask`.

Serving a Random Forest model will be similar to serving a Neural Network, so you might want to read [this](https://github.com/Matleo/MLPython2Java/tree/develop/Maschine%20Learning/NeuralNetwork/Serving), before continuing, to make yourself familiar with the `flask` framework.

## Training
Firstly you will want to get the MNIST dataset, which i will not get into too much, but for my [example](https://github.com/Matleo/MLPython2Java/blob/develop/Maschine%20Learning/RandomForest/IaaS/train.py) this happens in the`load_mnist()` function, which uses the `fetch_mldata()` function from `sklearn.datasets`:
```python
	train_data, test_data = load_mnist(10000) #size of test_data is 10000
```

Afterwards we can simply instantiate our `RandomForestClassifier` with:
```python
	clf = RandomForestClassifier(n_estimators=5, min_samples_split=50)
```
* `n_estimators`: Defines how many `DecicionTrees` should be used for the forest
* `min_samples_split`: The minimum number of data points per leaf required to split that leaf again

Fitting the forest is just as easy:
```python
	clf.fit(train_data["data"], train_data["target"])
```

After fitting the forest you can use the classifier to predict a new image, either calling the `predict()` function, getting the predicted number of the image, or calling the `predict_proba()` function which returns the score array, containing the probabilities for all numbers:
```python
	#pngArray = read_array_pixel_values_from_png(...)
	prediction = clf.predict(pngArray)
	score = clf.predict_proba(pngArray)
```

## Saving
Saving a `sklearn` Random Forest can easily be done by using pickle, which is Pythons object serialization module:
```python
	pickle.dump(clf, "export.pkl")
```
This will save a full serialized representation of the instantiated `RandomForestClassifier` to the "export.pkl" file, created in the current working directory. 

You can easily reload from this file aswell:
```python
	clf = pickle.load("export.pkl")
```

Since serializing a big Random Forest can take a long time, I used the `sklearn.externals.joblib` for my example, which uses `pickle` itself, but can serialize `numpy` arrays quicker. You can use this module just like `pickle`:
```python
	joblib.dump(clf, "export.pkl") #save
	...
	clf = joblib.load("export.pkl") #load
```

## Serving
I have designed this program in a way, that it uses the same API as if you would serve any of the [Neural Networks](https://github.com/Matleo/MLPython2Java/tree/develop/Maschine%20Learning/NeuralNetwork/Serving). Therefore you can either serve one of the Neural Networks or this Random Forest and use the same [InferenceClient](https://github.com/Matleo/MLPython2Java/blob/develop/MaschineLearning4J/src/main/java/InferenceClient.java) to send the grayscale pixel values of a png file as a 2-D integer array and get a prediction for the number that is displayed by that png file.


### Load the model
Firstly you will want to reload the pickled `RandomForestClassifier` and store it into the `app.config`, so that you can acces it later:
```python
	clf = joblib.load("export.pkl")
	app.config["clf"] = clf
```
For my example, I additionally store some meta data about the classifier in the configurations:
```python
	allParams = clf.get_params(False)
	params = {}
	params["_modelType"] = clf.__class__.__name__
	params["min_samples_split"] = allParams["min_samples_split"]
	params["n_estimators"] = allParams["n_estimators"]
	params["criterion"] = allParams["criterion"]

	train_data, test_data = load_mnist(10000)
	params["_accuracy"] = clf.score(test_data["data"], test_data["target"])

    app.config["modelMetaData"] = params
```

### Build a RESTful API

My example apllication will again respond to two routes:

* When calling `localhost:8000/`, you will get a JSON returned, representing the prediction for the Data/Own_dat/MNIST-0.png file. This route is used for demonstration purposes only.
* The actually important route is `localhost:8000/predict`, which is used for the actual prediction request. 
#### How /predict works: 
This route will expect a JSON as request parameter, containing a 2-dimensional integer array, representing the pixel values of a grayscale picture. Inside the 2D array, each array in the second dimension will contain the pixel information of a *row* of the picture (so the array will be of shape: [heigth][length])

##### Get request parameter
After getting the JSON array from the request, you will need to change it's datatype, because our image processing module `cv2` cannot handle the default of `np.int32`:
```python
	@app.route("/predict", methods=['POST'])
	def predict():
		json = request.get_json()
        reqArray = json["picArray"]
		pngArray = np.array(reqArray).astype(np.uint8)
```

##### Reshape parameters
Afterwards you will want to reshape the array, in order to fit into the pretrained classifier. Our classifier expects a batch of 784-vectors, where each value is a floating point in (0,1), where 0 represents white and 1 represents black. So the reshaping consists of 4 steps:
1. Resize the array to 28x28: 
```python
	reshaped_Array = cv2.resize(pngArray, (28, 28))
```
2. Linearly convert the pixel values from int(0,255) to float(0,1), where 0->1 and 255->0:
```python
	reshaped_Array = 1 - reshaped_Array / 255
```
3. Flatten the 2-dimensional 28x28 array to a 784-vector:
```python
	reshaped_Array = reshaped_Array.flatten()
```
4. Put the reshaped array in a batch of size 1:
```python
	reshaped_Array = [reshaped_Array]
```

##### Return prediction
Now we can use the prepared input parameter (`reshaped_array`) for prediction, using the previously loaded Random Forest (`clf`). As shown earlier, you can make use of the functions `predict()` and `predict_proba()`:
```python
	prediction = int(clf.predict(pngArray)[0])
	score = clf.predict_proba(pngArray)[0]
	predProb = round(score[prediction] * 100, 2)
```
As a different classifier could return more than one output value as the result of a prediction, the `sklearn` functions will always return a list of values. Since we know that our model will only return one output value, we can just access the first value of the list. 

Finally we can send a HTTP response to the calling client, containing the meta information about the classifier and the prediction results as a JSON:
```python
	params = app.config["modelMetaData"]
	return jsonify(_modelMetaData=params, _prediction=prediction, _probability=predProb)
```
This resulting JSON will look something like:
```json
{
  "_modelMetaData": {
    "_accuracy": 0.9202, 
    "_modelType": "RandomForestClassifier", 
    "criterion": "gini", 
    "min_samples_split": 50, 
    "n_estimators": 5
  }, 
  "_prediction": 5, 
  "_probability": 70.97
}
```

### Serve the RESTful API
To start the `flask` application and serve it to `localhost:8000`, use:
```python
	app = Flask(__name__)
	app.run(debug=True, port=8000)
```

For an example client that uses this prediction service, refer to [this](https://github.com/Matleo/MLPython2Java/tree/develop/MaschineLearning4J/src/main/java).

