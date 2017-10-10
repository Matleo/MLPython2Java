package NeuralNetwork.DL4J;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.modelimport.keras.UnsupportedKerasConfigurationException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.IOException;

public class IMDBClassifier {

    public static final int outputNum = 10;
    public static int batchSize = 128;
    public static int rngSeed = 123; //"random" seed

    public static void main(String args[]) throws IOException, InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {

        String importFile = "..\\Maschine Learning\\NeuralNetwork\\Keras\\IMDBClassifier\\export\\model.h5";

        //Dropout on LSTM not supported
        //LSTM doesnt work in dl4J 0.9.1 but 0.9.2 should fix this
        ComputationGraph network = KerasModelImport.importKerasModelAndWeights(importFile,false);

        int[] example = {15,256, 4, 2, 7, 3766, 5,723, 36, 71, 43,530
                ,476, 26,400,317, 46, 7, 4, 12118, 1029, 13,104, 88
                ,4,381, 15,297, 98, 32, 2071, 56, 26,141, 6,194
                ,7486, 18, 4,226, 22, 21,134,476, 26,480, 5,144
                ,30, 5535, 18, 51, 36, 28,224, 92, 25,104, 4,226
                ,65, 16, 38, 1334, 88, 12, 16,283, 5, 16, 4472,113
                ,103, 32, 15, 16, 5345, 19,178, 32};
        INDArray input = Nd4j.create(new int[]{0},example);
        INDArray output = network.output(input)[0];

        System.out.println(output);
    }

}
