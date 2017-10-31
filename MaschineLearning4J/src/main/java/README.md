# Java Side
## Model as a Service
For details on how to fully reload a previously built and trained model from Python into Java, please refer to the [NeuralNetwork](https://github.com/Matleo/MLPython2Java/tree/develop/MaschineLearning4J/src/main/java/NeuralNetwork) or [RandomForest](https://github.com/Matleo/MLPython2Java/tree/develop/MaschineLearning4J/src/main/java/RandomForest) directory.
## Inference as a Service
After you have built and served a RESTful API for making predictions of given images (as done [here](https://github.com/Matleo/MLPython2Java/blob/develop/Maschine%20Learning/NeuralNetwork/Serving) or [here](https://github.com/Matleo/MLPython2Java/tree/develop/Maschine%20Learning/RandomForest/serve.py)), you can use the [InferenceClient.java](https://github.com/Matleo/MLPython2Java/blob/develop/MaschineLearning4J/src/main/java/InferenceClient.java) to send such a request. This will post a 2-dimensional integer array, representing the grayscale pixels of an image and get the prediction of what number is most likely displayed as response. Notice that you can either serve a Neural Network or the Random Forest example. Both will respond the same way to this request and on the same URL.
### Dependency
Firstly, you will have to add the `apache httpclient` and `json-simple` dependency to your pom.xml:
```maven
	<dependency>
		<groupId>org.apache.httpcomponents</groupId>
		<artifactId>httpclient</artifactId>
		<version>4.5.2</version>
	</dependency>
	<dependency>
		<groupId>com.googlecode.json-simple</groupId>
		<artifactId>json-simple</artifactId>
		<version>1.1.1</version>
	</dependency>
```
### Usage


As Program parameter you can pass in either an absolute path name to a .png file, or the name of a .png located inside the [Data/Own_dat](https://github.com/Matleo/MLPython2Java/tree/develop/Maschine%20Learning/Data/Own_dat) folder. If you do not pass any parameter, the prediction request is going to be sent with a default picture. 

If the service is running correctly, the output will look something like this :
```java
	Addressing the webservice with default picture: ../Maschine Learning/Data/Own_dat/MNIST-5.png

	Sent the request, waiting for a response...
	Received the response.

	Response status line: HTTP/1.0 200 OK
	Response content: {
		"_modelMetaData": {
			"_accuracy": 0.9749, 
			"_modelType": "Estimator Convolutional Neural Network", 
			"batch_size": 50, 
			"steps": 200
		}, 
		"_prediction": 0, 
		"_probability": 100.0
}
``` 
Notice that the "Response content" will slightly vary, depending on which model you are currently serving. In the above example i was using my example CNN, built with `Tensorflow.Estimator`. 

Response content interpretation: 
* `_modelMetaData`: will be present for every model, containing model describing paramers
	* `_accuracy`: will be present for every model and describes the accuracy of the model on the test MNIST test data
	* `_modelType`: will also be present for every model, describing the type of model
	* `batch_size`: is specific to Tensorflow or Estimator Neural Networks and describes the size of one batch, used for training
	* `steps`: is also specific to Tensorflow or Estimator Neural Networks and describes the amount of batches used for training
* `_prediction`: will be present for every model, containing the actual predicted number for the sent picture
* `_probability`: will also be present for every model and describes how certain the model prediction is
### How it works
After evaluating the program parameters, the program works as follows:
1. Read the grayscale pixels from a given picture into a 2D integer array (i have omitted the try-catch blocks for better readability):
```java
	File imgFile = new File(path); //"path" is the filepath to the .png
	BufferedImage img = ImageIO.read(imgFile);
	Raster raster = img.getData();
	int[][] imgArrInt = new int[img.getHeight][img.getWidth];
 	for (int i = 0; i < img.getHeigth(); i++) {
		for (int j = 0; j < img.getWidth(); j++) {
			imgArrInt[i][j] = raster.getSample(j, i, 0);
		}
	}
	return imgArrInt;
```
*Note*: `raster.getSample(j, i, 0)` returns the value of the pixel at width=j and heigth=i. As we want to return an array of shape [heigth][width], the indices to store into the int[][] need to be reversed.  

2. Convert the `int[][]` to a JSON array within a JSON object to pass it to the HTTP request, using the `json-simple` library:
```java
	JSONObject json = new JSONObject();
	JSONArray picArray = new JSONArray();
	for (int i = 0; i < picture.length; i++) { //"picture" is the int[][] here
		JSONArray subArr = new JSONArray();
		for (int j = 0; j < picture[i].length; j++) {
			subArr.add(picture[i][j]);
		}
		picArray.add(subArr);
	}
	json.put("picArray", picArray);
	String JSON_STRING = json.toJSONString();
```

3. Create the HTTP Post request with the JSON as parameter, using the `apache httpclient` library:
```java
	CloseableHttpClient httpclient = HttpClients.createDefault();
	HttpPost httpPost = new HttpPost(Server_URI); //"Server_URI" describes the full URI where to send the request to
	StringEntity requestEntity = new StringEntity(JSON_STRING, ContentType.APPLICATION_JSON);
	httpPost.setEntity(requestEntity);
```

4. Send the HTTP Post request to the RESTful API and receive the response:
```java
	CloseableHttpResponse response = httpclient.execute(httpPost);
```

5. Read out the HTTP response:
```java
 	System.out.println("Response status line: " + response.getStatusLine());
 	HttpEntity entity = response.getEntity();
 	InputStream is = entity.getContent();
 	String content = IOUtils.toString(is, StandardCharsets.UTF_8);
 	System.out.println("Response content: " + content);
```