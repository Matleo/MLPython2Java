package NeuralNetwork.Tensorflow;

import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

import java.util.List;


/**
 * class to load and store the trained Tensorflow Model
 */
public class SavedModel {
    private Session s;
    private static final String input = "input:0";
    private static final String output = "output:0";
    private static final String kerasOutput = "Softmax:0";
    private static final String kerasLearningPhase = "keras_learning_phase";
    private static final String dropout = "dropoutRate:0";


    public SavedModel(String importDir, String ModelTag) {
        SavedModelBundle sb = SavedModelBundle.load(importDir, ModelTag);
        this.s = sb.session();
    }

    /**
     * Function to predict the Number from a read File using the loaded NN Model
     *
     * @param inputArray transformed float Array, which represents the read File/Picture
     * @param modelType  which modeltype (low level tensorflow, keras, estimator)
     * @return the predicted (most likely to be) number as an integer
     */
    protected int predictNumber(float[] inputArray, String modelType) {
        Tensor result = getOutput(inputArray,modelType);
        return TensorflowUtilities.maxIndex(TensorflowUtilities.toArray(result));
    }

    protected Tensor getOutput(float[] inputArray, String modelType){
        Tensor inputTensor = TensorflowUtilities.toTensor(inputArray);
        Tensor dKeep = Tensor.create(1f);
        Tensor learningPhase = Tensor.create(false);

        List<Tensor> resultList;
        if (modelType== "Tensorflow") {
            resultList = this.s.runner().feed(input, inputTensor).feed(dropout, dKeep).fetch(output).run();
        } else if(modelType=="Estimator"){
            resultList = this.s.runner().feed(input, inputTensor).fetch(output).run();
        }else {//KerasModel or KerasSequential
            resultList = this.s.runner().feed(input, inputTensor).feed(kerasLearningPhase,learningPhase).fetch(kerasOutput).run();
        }

        Tensor result = resultList.get(0);
        return result;
    }


}
