package NeuralNetwork.DL4J;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.modelimport.keras.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.IOException;

/**
 * Class to load a pre-trained Keras Sequential model and test it against the MNIST test data
 * comments:
 * Keras import generaly only works until keras v1.2.2 (might change in dl4j 0.9.1)
 * Reshape and LSTM layer not supportet in dl4j 0.9.1 but will come with 0.9.2
 */
public class MNISTClassifier {
    public static final int outputNum = 10;
    public static int batchSize = 128;
    public static int rngSeed = 123; //"random" seed
    public static String importFile1 = "..\\Maschine Learning\\NeuralNetwork\\Keras\\MNISTClassifier\\Model\\export\\my_model.h5";
    public static String importFile2 = "..\\Maschine Learning\\NeuralNetwork\\Keras\\MNISTClassifier\\Sequential\\export\\my_model.h5";


    public static void main(String[] args) throws UnsupportedKerasConfigurationException, IOException, InvalidKerasConfigurationException {
        if (args.length > 0) {
            if ((args[0].equals("ffnn"))) {
                launchSequentialDense();
            } else if(args[0].equals("cnn")){
                launchModelCNN();
            }else{
                System.out.println("You passed in an invalid parameter");
                System.out.println("arg: "+args[0]);
            }
        }else{
            System.out.println("Please pass a program parameter, to define whether to launch the saved CNN or FFNN");
            System.out.println("Pass \"ffnn\" for the feed forward neural network");
            System.out.println("Pass \"cnn\" for the convolutional neural network");
        }
    }

    public static void launchSequentialDense() throws UnsupportedKerasConfigurationException, IOException, InvalidKerasConfigurationException {
        String importFile = importFile2;

        MultiLayerNetwork network = KerasModelImport.importKerasSequentialModelAndWeights(importFile);

        Evaluation evaluator = new Evaluation(outputNum); //create an evaluation object with 10 possible classes

        DataSetIterator mnistTest = new MnistDataSetIterator(batchSize, false, rngSeed);

        while (mnistTest.hasNext()) {
            DataSet next = mnistTest.next();
            INDArray output = network.output(next.getFeatureMatrix()); //get the networks prediction
            evaluator.eval(next.getLabels(), output); //check the prediction against the true class
        }

        //success: evaluates the exact same accuracy value as python Keras
        System.out.println(evaluator.stats());
    }


    public static void launchModelCNN() throws UnsupportedKerasConfigurationException, IOException, InvalidKerasConfigurationException {
        String importFile = importFile1;

        //works only until keras v1.2.2
        ComputationGraph network = KerasModelImport.importKerasModelAndWeights(importFile);

        Evaluation evaluator = new Evaluation(outputNum); //create an evaluation object with 10 possible classes

        DataSetIterator mnistTest = new MnistDataSetIterator(batchSize, false, rngSeed);
        while (mnistTest.hasNext()) {
            DataSet next = mnistTest.next();
            INDArray batch = next.getFeatureMatrix();
            INDArray reshapedBatch = batch.reshape(batch.shape()[0], 1, 28, 28);//reshape for convolution layer (because dl4j0.9.1 doesnt support Keras.Reshape()
            INDArray output = network.output(reshapedBatch)[0]; //get the networks prediction
            evaluator.eval(next.getLabels(), output); //check the prediction against the true class
        }

        //success: evaluates the exact same accuracy value as python Keras
        System.out.println(evaluator.stats());
    }
}
