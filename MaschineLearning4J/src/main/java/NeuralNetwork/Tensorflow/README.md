# Using a saved Tensorflow SavedModel
In the following i will describe, how to use a model for inference that was previously built and trained either with the `Tensorflow`, `Estimator` or `Keras` API and later saved as a `Tensorflow SavedModel`. Before reading this part, you might want to read about how to actually save a model, which is described at the [Python part](https://github.com/Matleo/MLPython2Java/tree/develop/Maschine%20Learning/NeuralNetwork).


## Project setup
For this project i am using the `Tensorflow Java`, the `Coobird Thumbnailator` and the `Googlecode Json-Simple` libraries. To add these to your maven project, simply add these dependencies to your `pom.xml`:
```xml
        <dependency>
            <groupId>org.tensorflow</groupId>
            <artifactId>tensorflow</artifactId>
            <version>1.3.0</version>
        </dependency>
        <dependency>
            <groupId>net.coobird</groupId>
            <artifactId>thumbnailator</artifactId>
            <version>0.4.8</version>
        </dependency>
        <dependency>
            <groupId>com.googlecode.json-simple</groupId>
            <artifactId>json-simple</artifactId>
            <version>1.1.1</version>
        </dependency>
```

### Class structure
To make use of a `SavedModel` i created three Java classes:
#### MNISTClassifier.java:
The main class, to execute either loading the whole model and making a prediction using the Tensorflow Java API, or send a request to receive the inference from a service.   
Depending on the program parameter that you pass, different tasks will be executed. With passing "-h" you can print this help again:

Valid arguments include:
    
* `-m|-model <value>`: Specify which SavedModel to use for predictions. `<value>´ can be any of "t_ffnn" / "t_cnn" / "e_ffnn" / "e_cnn" / "k_ffnn" / "k_cnn". If you don't pass this argument, the default Tensorflow Feed Forward Neural Network will be loaded
* `--noEval`: Pass if the imported model should **not** be evaluated against the saved results from the original Python model. If you pass this argument, you need to specify `-p`
* `-p|-predict <path>`: Specify of which picture to predict its number. `<path>` needs to be a absolute path and pointing to a .png file. If you don't pass this argument, no prediction will be executed.

#### SavedModel.java:
This class implements the *Model as a Service* idea. Here we are actually using the Tensorflow Java API, to reload a `SavedModel`. Afterwards we can make predictions using the loaded model and evaluate if the model produces just the same results, as if we would use the original Python model, by comparing the accuracy on the MNIST dataset and the predictions of a bunch of saved .png files.

#### TensorflowUtilities.java:
This class contains a few utility functions to be used by the other two classes, like converting a `Tensor` to a `float[]` and the other way around, reading a .png file into a `float[]` or providing an equivalent funtion to the `numpy.argmax()`.
                            
## Model as a Service

As stated, the whole action of loading and using a `SavedModel` to make predictions happens in the `SavedModel.java` class, using the Tensorflow Java API.

The only thing we really need to load from the `SavedModel` is it's `Session`. After loading the `Session` we can start using it to make predictions:
```java
        SavedModelBundle sb = SavedModelBundle.load(importDir, ModelTag);
        this.session = sb.session();
```
* `importdir` is a `String`, containing the path of the directory of where to load the `SavedModel` from
* `modelTag` is a `String`, containing the tag of the appropriate `MetaGraph`, which we want to load from the `SavedModel`. In my example, this will always be the constant String `"serve"`, which i used to save the `SavedModels`.

Now, to use the loaded `Session` for inference, let me remind you, that it is very important to know the exact names of the input and output tensors inside the `SavedModel`. Depending on which technology you used to save your `SavedModel`, you where either able specify these names explicitly (`Tensorflow` or `Estimator`), or if you used `Keras`, you will have to inspect the `SavedModel` and find the corresponding names, using the [SavedModel CLI](https://www.tensorflow.org/programmers_guide/saved_model#cli_to_inspect_and_execute_savedmodel).

In my examples, the important `Tensor` names look as following:
```java
    #Tensorflow or Estimator:
	private static final String input = "input:0";
    private static final String output = "output:0";
    private static final String dropout = "dropoutRate:0";
	
	#Keras:
    private static final String kerasInput = "input_input:0";
    private static final String kerasOutput = "output/Softmax:0";
    private static final String kerasLearningPhase = "dropout_1/keras_learning_phase:0";
```
Note that the names for a `Keras` model differ, and that i set the names of the `Estimator` or `Tensorflow` models identically. 

After loading a selected image into a `float[]`, you can load that array into a `Tensor`, feed it to the `Session` and fetch the output `Tensor`:
```java
        float[] inputArray = TensorflowUtilities.readPic(pathfile);
        Tensor inputTensor = TensorflowUtilities.toTensor(inputArray); #create Tensor from float[]
        
        List<Tensor> resultList = this.session.runner().feed(input, inputTensor).fetch(output).run();

```
Afterwards you can read the `resultList`, which will contain the corresponding score/probabilities for the prediction of the given image.

Depending on the model that you saved, you will have to feed different, multiple input `Tensors`, and for the `Keras` models, the names of the `Tensors` need to be adapted.
```java
    #my Tensorflow models
    Tensor dKeep = Tensor.create(1f);
    resultList = this.session.runner().feed(input, inputTensor).feed(dropout, dKeep).fetch(output).run();

    #my Keras models
    Tensor learningPhase = Tensor.create(false);
    resultList = this.session.runner().feed(kerasInput, inputTensor).feed(kerasLearningPhase, learningPhase).fetch(kerasOutput).run();   
```
## Inference as a Service
**TODO**