# Keras
Keras is a high-level neural networks API, written in Python and capable of running on top of TensorFlow. Keras provides two APIs to build a neural network, the `Sequential` and the `Model`. Basically you should use a `Sequential` if your model is simple and essentially just a sequence of layers. If you are trying to build a more complex models, which includes non-sequential connections and multiple inputs/outputs, you should use the functional `Model` API.

I used a `Sequential` to create my [FFNN example](https://github.com/Matleo/MLPython2Java/tree/develop/Maschine%20Learning/NeuralNetwork/Keras/MNISTClassifier/Sequential) and a `Model` to create my [CNN example](https://github.com/Matleo/MLPython2Java/tree/develop/Maschine%20Learning/NeuralNetwork/Keras/MNISTClassifier/Sequential). Additionally i created a [LSTM example](https://github.com/Matleo/MLPython2Java/tree/develop/Maschine%20Learning/NeuralNetwork/Keras/IMDBClassifier) for IMDB Classification to validate that the export works with complex models aswell.

I will assume that you are familiar with building and training a Keras model, if that is not the case, please refer to the original [documentation](https://keras.io/). 

## Model as a Service
I evaluated two seperate options to export a Keras model from Python and reload it into Java:
1. Save the native Keras model into a .h5 file and reload it with the [DL4J](https://deeplearning4j.org/) framework.
2. Get the Tensorflow `session` from the model and save it as a `SavedModel`
### Native Keras
Before saving your model, make sure that you are on Keras 1.2.2 or lower, as ND4J v.0.9.1 does not support higher versions. This compatibility is going to change with time though.

Saving the native Keras model is as easy as:
```python
    model.save("./export/my_model.h5")
```
and works the same with a Keras `Model` as with a Keras `Sequential`.

Loading the model into Python is just as easy:
```python
    model = keras.models.load_model("./export/my_model.h5")
```


The interesting part happes at the [Java part](https://github.com/Matleo/MLPython2Java/tree/develop/MaschineLearning4J/src/main/java/NeuralNetwork/DL4J). 

It is important to note, that you can **not** use every Keras element, as the Java import with DL4J only supports a restricted amount of [features](https://deeplearning4j.org/keras-supported-features). This list of features will be available with DL4J v.0.9.2 which is not stable at the moment. I used DL4J v.0.9.1, where there were fewer options available. Because of this reason, and because using an additional (small) framework can often be unsafe, i reccomend using the second method, to extract the Tensorflow `session` and export a `SavedModel`.

You can find my full example for a `Sequential` [here](https://github.com/Matleo/MLPython2Java/blob/develop/Maschine%20Learning/NeuralNetwork/Keras/MNISTClassifier/Sequential/train_dl4j.py), and for a `Model` [here](https://github.com/Matleo/MLPython2Java/blob/develop/Maschine%20Learning/NeuralNetwork/Keras/MNISTClassifier/Model/cnn_train_dl4j.py). As i was not able to use  all the Keras layers here, i seperated exporting the native Keras model and the Tensorflow `SavedModel` in seperate files. (*Note*: The LSTM example doesn't work with DL4J on v0.9.1)

### Exporting a SavedModel
As mentioned, Keras is a model-level library, providing high-level building blocks for developing deep learning models. It does not handle low-level operations such as tensor products, convolutions and so on itself. Therefore it calls the Tensorflow API to build a `Graph` and run a `Session`. We can access the `Session` by using the `Keras.backend`, which contains the backend engine, in our case `Tensorflow`: 
```python
		#build and train model...
		from keras import backend as K
        sess = K.get_session()
        graph = sess.graph
```

Now we can export a `SavedModel`  as usual. Some Keras layers (e.g. Dropout, BatchNormalization) behave differently at training time and testing time. If you are using one of these layers, you will need to feed a boolean value for the `keras_learning_phase` additionally to the usual 784 input vector. Keras itself sets the appropriate values automatically, but for the model to be used by the Tensorflow Java API, we will need to feed the value explicitly. Therefore we will adapt the signature:
```python
    learningPhase = K.learning_phase()
    signature = tf.saved_model.signature_def_utils.build_signature_def(
        		inputs={'input': tf.saved_model.utils.build_tensor_info(model.input),
                		'keras_learning_phase': tf.saved_model.utils.build_tensor_info(learningPhase)},
        		outputs={'output': tf.saved_model.utils.build_tensor_info(model.output)})
    signatureDef = {tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature}
```

The execution of the export works just as usual:
```python
	builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
    builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING], signature_def_map=signatureDef)
    builder.save()
```

Just to be clear, adding the signature is not mandatory, everything will work just the same if you don't (as long as you know the names of the tensors). But if that is not the case it is very usefull to inspect the `SavedModel` using the [SavedModel CLI](https://www.tensorflow.org/programmers_guide/saved_model#cli_to_inspect_and_execute_savedmodel). You will need to know the exact names of the input and output Tensors to reuse the model in Java.

Talking about naming the tensors. I did not find a way to rename the tensorflow input and output tensors using the Keras API, this means that we will have to work with the default names that Tensorflow assigns to these Tensor. The only tensor i was able to rename is the input tensor of a Keras `Model`, since you can specify the input as an `Input` layer:
```python
    inputs = Input(shape=(784,), name="input_input")
```
Note that i used the same name here as the name assigned to the input of my `Sequential` example. So that the signature of my `Model` [example](https://github.com/Matleo/MLPython2Java/tree/develop/Maschine%20Learning/NeuralNetwork/Keras/MNISTClassifier/Model) and my `Sequential` [example](https://github.com/Matleo/MLPython2Java/tree/develop/Maschine%20Learning/NeuralNetwork/Keras/MNISTClassifier/Sequential) are identical.

With the default tensor names, this is what the signature of our `SavedModel` looks like:
![SavedModelCLI output picture](https://github.com/Matleo/MLPython2Java/tree/develop/Maschine%20Learning/NeuralNetwork/Keras/MNISTClassifier/SavedModelCLI.png)

## Inference as a Service
**TODO**