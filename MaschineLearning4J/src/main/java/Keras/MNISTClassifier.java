package Keras;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.modelimport.keras.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.modelimport.keras.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.IOException;

/**
 *  Class to load a pre-trained Keras Sequential model and test it against the MNIST test data
 */
public class MNISTClassifier {
    public static void main(String[] args) throws UnsupportedKerasConfigurationException, IOException, InvalidKerasConfigurationException {
        final int outputNum = 10;
        final int batchSize = 128;
        final int rngSeed = 123; //"random" seed
        final String importFile = "..\\Maschine Learning\\NeuralNetwork\\Keras\\MNISTClassifier\\export\\my_model.h5";
        //Todo: slf4j Logger


        //works only until keras v1.2.2
        MultiLayerNetwork network = KerasModelImport.importKerasSequentialModelAndWeights(importFile);

        Evaluation evaluator = new Evaluation(outputNum); //create an evaluation object with 10 possible classes

        DataSetIterator mnistTest = new MnistDataSetIterator(batchSize, false, rngSeed);

        while(mnistTest.hasNext()){
            DataSet next = mnistTest.next();
            INDArray output = network.output(next.getFeatureMatrix()); //get the networks prediction
            evaluator.eval(next.getLabels(), output); //check the prediction against the true class
        }

        //success: evaluates the exact same accuracy value as python Keras
        System.out.println(evaluator.stats());

    }
}
