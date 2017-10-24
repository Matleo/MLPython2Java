# Neural Networks Java
In this part of the project i will describe how to use a model that was previously built, trained and exported in Python. If you have not looked at the [Python side](https://github.com/Matleo/MLPython2Java/tree/develop/Maschine%20Learning/NeuralNetwork) yet, i recommend doing so.

## Model as a Service
I used two seperate methods to export a Python machine learning model:
1. Keras only: Save the model into a .h5 file
2. Save the model as a Tensorflow `SavedModel`

In the [DL4J](https://github.com/Matleo/MLPython2Java/tree/develop/MaschineLearning4J/src/main/java/NeuralNetwork/DL4J) subdirecory, you will find the code to reload a previously saved Keras model with the [DL4J framework](https://deeplearning4j.org/). 

Whereas in the [Tensorflow](https://github.com/Matleo/MLPython2Java/tree/develop/MaschineLearning4J/src/main/java/NeuralNetwork/Tensorflow) subdirecory, you will find the code using the [Java Tensorflow API](https://www.tensorflow.org/api_docs/java/reference/org/tensorflow/package-summary) to import a `SavedModel`.

## Inference as a Service
After you have built and served a RESTful API for making predictions of given images (as done [here](https://github.com/Matleo/MLPython2Java/blob/develop/Maschine%20Learning/NeuralNetwork/Serving)), you can use the [InferenceClient.java](https://github.com/Matleo/MLPython2Java/blob/develop/MaschineLearning4J/src/main/java/NeuralNetwork/InferenceClient.java) to send such a request. This will post a 2-dimensional integer array, representing the grayscale pixels of an image and get the prediction of what number is most likely displayed as response.
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

If the service is running correctly, the output will look something like:
```java
	Infering the webservice with default picture: ../Maschine 							Learning/Data/Own_dat/MNIST-5.png
	Response status line: HTTP/1.1 200 OK
	Response content: {
		"prediction": 5, 
		"probability": 90.8744752407074
	}
``` 
### How it works
After evaluating the program parameters, the program works as follows:
1. Read the grayscale pixels from a given picture into a 2D integer array (i have omitted the try-catch blocks for readability):
```java
	File imgFile = new File(path); //"path" is the filepath to the .png
	BufferedImage img = ImageIO.read(imgFile);
	Raster raster = img.getData();
    imgArrInt = new int[img.getHeight][img.getWidth];
 	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			imgArrInt[i][j] = raster.getSample(j, i, 0);
		}
	}
    return imgArrInt;
```
*Note*: `raster.getSample(j, i, 0)` returns the value of the pixel at width=j and heigth=i. As we want to return an array of shape [heigth][width], the indices needs to be reversed.  

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
	CloseableHttpResponse response = null;
	response = httpclient.execute(httpPost);
```

5. Read out the HTTP response:
```java
 	System.out.println("Response status line: " + response.getStatusLine());
 	HttpEntity entity = response.getEntity();
 	InputStream is = entity.getContent();
 	String content = IOUtils.toString(is, StandardCharsets.UTF_8);
 	System.out.println("Response content: " + content);
```