# Low level Tensorflow API
I worked with Tensorflow version 1.3, the full documentation can be found [here](https://www.tensorflow.org/api_docs/).

I am going to focus on describing the workflow to save a Tensorflow model, rather then explaining how the building and training works. If you are looking for an introduction on the MNIST dataset and Tensorflow syntax, you might want to read [Tensorflow getting started](https://www.tensorflow.org/get_started/mnist/beginners) and take a look at my commented code [here](https://github.com/Matleo/MLPython2Java/blob/develop/Maschine%20Learning/NeuralNetwork/Tensorflow/MNISTClassifier/Feed%20Forward%20NN/saver/NN_learn.py).

## Installation
Please follow the official [instructions](https://www.tensorflow.org/install/install_windows)

Furthermore you will need:
* **TODO**

## Model as a Service
### Saver
First i tried to save the trained model in a format, so that i can save it to a file and reimport it later in python. This was easily accomplished by using the `tensorflow.train.Saver()`. 
#### Export
The important code snippet from the [export code](https://github.com/Matleo/MLPython2Java/blob/develop/Maschine%20Learning/NeuralNetwork/Tensorflow/MNISTClassifier/Feed%20Forward%20NN/saver/NN_learn.py) is:
```python
    x = tf.placeholder(tf.float32, [None, 784], name="input")
    dKeep = tf.placeholder(tf.float32, name="dropoutRate")
    #build some layers...
    y3 = tf.nn.softmax(tf.matmul(y2, W3) + b3)
    y3 = tf.identity(y3, "output")

    sess = tf.Session()
    #train the model...
    export_dir = "./export/model"
    saver = tf.train.Saver()
    saver.save(sess, export_dir)
```
Which creates multiple files in the sub directory "export" with the name "model".xxx. 
I explicitly showed how to create the input tensors `x` and `dKeep` and the output tensor `y3`, because we will need the names of these tensors later for the import. Note that the name of the output tensor needs to be changed, by recreating the tensor with a different name, using `tf.identity()`.

#### Import
The model, with all its weights and biases, can then be loaded into a newly created session:
```python
    import_dir = "./export/model"
    sess = tf.Session()
    saver = tf.train.import_meta_graph(import_dir + ".meta")
    saver.restore(sess, import_dir)
```
Where first, the model structure is loaded into the saver object`(tf.train.import_meta_graph())` from the model.meta file, before the model with it's weights and biases are loaded into the new session`(saver.restore())`.

To be able to make predictions with the imported model, you will need to grab the input and output tensors from the model, in order to feed/fetch values:
```python
    graph = tf.get_default_graph()
    y3 = graph.get_tensor_by_name("output:0")
    x = graph.get_tensor_by_name("input:0")
    dKeep = graph.get_tensor_by_name("dropoutRate:0")
    
    #actual prediction call (inputArray is an array, read from a png):
    prediction = sess.run(y3, feed_dict={x: inputArray, dKeep: 1})

```
The `tf.get_default_graph()` gets the actual graph of the imported session. Important to notice is that the names of the tensors need to match the names of the originally exported model (which is shown above). 

For more information you can see the example of the [import code](https://github.com/Matleo/MLPython2Java/blob/develop/Maschine%20Learning/NeuralNetwork/Tensorflow/MNISTClassifier/Feed%20Forward%20NN/saver/NN_test.py) and the full documentation of the [Tensorflow Saver](https://www.tensorflow.org/api_docs/python/tf/train/Saver#restore). 

Saving the Estimator and Keras Models are going to be based on *SavedModel* aswell. 
## Inference as a Service
**Todo**