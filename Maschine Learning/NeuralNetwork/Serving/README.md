# Serving a NN model with flask
After we created, trained and saved our model for MNIST classification, we can now start using it to provide a RESTful API. Before continuing here, please make sure that you have read about how to [save a SavedModel](https://github.com/Matleo/MLPython2Java/tree/develop/Maschine%20Learning/NeuralNetwork/Tensorflow/MNISTClassifier).

Running the [Flask_Serving](https://github.com/Matleo/MLPython2Java/blob/develop/Maschine%20Learning/NeuralNetwork/Serving/Flask_Serving.py) script will start an application, which loads a previously trained and saved model and provides a RESTful API at `localhost:8000`, using the [flask](http://flask.pocoo.org/) microframework.

## Loading the model
To specify, which model you want to load and serve, pass the script argument `--model` with a value in one of: `t_ffnn` , `t_cnn` , `e_ffnn` , `e_cnn` , `k_ffnn` , `k_cnn`. This input will be evaluated by the `getImportDir(model)` function, which will determine from which directory to load the `SavedModel`.

Loading one of the saved models happens in the `load_model(model)` function, which takes the value of the script argument as parameter. What this function does is nothing but reloading the previously exported `SavedModel` from the file system and storing it into the configuration of the flask application, so that the model can be accessed later. The actual reloading can be done in this one line of code:
```python
    tf.saved_model.loader.load(session, ["serve"], import_dir)
```
Afterwards, storing the session into the application configuration can be done by accessing the `app.config` dictionary:
```python
	app.config["modelType"] = model
    app.config["session"] = sess
    app.config["graph"] = graph
```

## Providing a RESTful API
My example apllication will respond to two routes: 
* When calling `localhost:8000/`, you will get a JSON returned, representing the prediction for the [Data/Own_dat/MNIST-0.png](https://github.com/Matleo/MLPython2Java/blob/develop/Maschine%20Learning/Data/Own_dat/MNIST-0.png) file, this route is only used for demonstration purposes. 
* The more important route is `localhost:8000/predict`, which is used for the actual prediction request. It will expect a JSON as request parameter, containing a 2-dimensional integer array, representing the pixel values of a grayscale picture. Inside the 2D array, each array in the second dimension should contain the pixel information of a row of the picture (so the array should be of shape: [heigth][length])

### Define application and route
With flask, you can create your application by calling the `Flask()` constructor:
```python
	app = Flask(__name__)
```
Afterwards you can add a new route, by adding a function decorator. The function will then be executed, if the assosiated route is called:
```python
	@app.route("/predict", methods=['POST'])
    def predict():
		...
```

A requests parameter can be accessed by using the `flask.request` object. We are expecting an array with the name "picArray" inside a JSON to be passed. We can acces the array with:
```python
    json = request.get_json()
    picArray = json["picArray"]
```

### Reshaping the input array
After receiving the sent array, we need to reshape it so that it fits into the model and we are able to get a prediction. The reshaping is handled by the `reshapePic(pic)` function.  

This step depends on the model you want to use. For the MNIST example, the input needs to be a batch of vectors of length 784. These vectors are the result of flattening the pixel information of a 28x28 array, where each pixel value is a float in [0,1], where 1 means black and 0 means white. 

The reshaping can be done, following these steps:
1. Wrap each value inside the array into another array. This needs to be done, because Tensorflow expects the input to be 3-dimensional, with channel last. This means, that the last dimension contains the values for each channels (red, green, blue) of the picture. For this example there is only 1 channel, because the pictures are grayscale).
2. Resize the array to 28x28
3. Flatten the array into a vector
4. Convert the value of the pixels. The value of a grayscale pixel is an integer in [0,255], where 255 means white and 0 means black. This is the opposite of what we need. Therefore we need to reverse the value and convert it into a float value in [0,1]
5. Make a batch of size 1, with the created vector as value.

### Making a prediction
Using the loaded model to make a prediction will depend on the architecture of the model that you are using. In the example, you can either use a model that was trained with `Tensorflow`, `Estimator` or `Keras`. 

Firstly, you will have to get the input and output tensors from the graph. For example if you are using an `Estimator` model:
```python
	y = graph.get_tensor_by_name("output:0")
    x = graph.get_tensor_by_name("input:0")
```
After that, you can run the model, feeding the previously received and reshaped array, representing a grayscale image. As output you will get the score of the classification probabilities. 
```python
    score = sess.run(y, feed_dict={x: picArray})[0]
```
Depending on which model you loaded, the actual prediction call will vary. For more information, take a look at the training and export of the [specific model](https://github.com/Matleo/MLPython2Java/tree/develop/Maschine%20Learning/NeuralNetwork) of your choice. For the example, the `predictPic()` function executes the appropriate prediction.

As we want to respond to the incoming request with a JSON, we can use the `flask.jsonify` function to wrap the calculated prediction for the given image and it's respective probability:
```python
    pred = np.argmax(score)
    predProb = score[pred] * 100

    return jsonify(prediction=int(pred), probability=float(predProb))
```

For an example client that uses this prediction service, refer to [this](https://github.com/Matleo/MLPython2Java/blob/develop/MaschineLearning4J/src/main/java/NeuralNetwork/InferenceClient.java).