# Keras + DL4J
This was my first approach to export a `Keras` model and import it into Java, using the `DL4J` framework. As explained [here](https://github.com/Matleo/MLPython2Java/tree/develop/Maschine%20Learning/NeuralNetwork/Keras), I do not recommend using `Keras` and `DL4J`, as there is the possibility to extract the `Tensorflow Session` from the `Keras` model and save it as a `SavedModel`. Anyways this might be useful at a later point in time.

### Prerequisites
Add the following dependencies to your `pom.xml`:
```xml
    <dependency>
        <groupId>org.deeplearning4j</groupId>
        <artifactId>deeplearning4j-core</artifactId>
        <version>${dl4j-version}</version>
    </dependency>
    <dependency>
        <groupId>org.deeplearning4j</groupId>
        <artifactId>deeplearning4j-nlp</artifactId>
        <version>${dl4j-version}</version>
    </dependency>
    <dependency>
        <groupId>org.deeplearning4j</groupId>
        <artifactId>deeplearning4j-modelimport</artifactId>
        <version>${dl4j-version}</version>
    </dependency>
    <dependency>
        <groupId>org.nd4j</groupId>
        <artifactId>nd4j-native-platform</artifactId>
        <version>0.9.1</version>
    </dependency>
```

### Import
After defining the directory in which to find the saved .h5 file, importing a `Keras Sequential` is done with this one-liner:
```java
    MultiLayerNetwork network = KerasModelImport.importKerasSequentialModelAndWeights(importFile);

```

Importing a `Model` is just a little bit different:
```java
    ComputationGraph network = KerasModelImport.importKerasModelAndWeights(importFile);

```

### Using the model
After importing the Keras `Model` or `Sequential` you can use the DL4J [API](https://deeplearning4j.org/overview) as usual. 

For example you can load the MNIST dataset and evaluate the accuracy with the loaded DL4J model:
```java
    Evaluation evaluator = new Evaluation(outputNum);

    DataSetIterator mnistTest = new MnistDataSetIterator(batchSize, false, rngSeed);

    while (mnistTest.hasNext()) {
        DataSet next = mnistTest.next();
        INDArray output = network.output(next.getFeatureMatrix());
        evaluator.eval(next.getLabels(), output);
    }

    System.out.println(evaluator.stats());
```
As you will see, the accuracy matches the in Python computed accuracy. 

For more information on how to load a Keras model with DL4J please refer to [this](https://deeplearning4j.org/model-import-keras). 