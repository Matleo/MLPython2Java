package Tensorflow;

import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

import java.util.List;

import static Tensorflow.TensorflowUtilities.*;


/**
 * class to load and store the trained Tensorflow Model
 */
public class SavedModel {
    private Session s;
    private static final String input = "input:0";
    private static final String tensorflowOutput = "output:0";
    private static final String estimatorOutput = "dnn/head/predictions/probabilities:0";
    private static final String dropout = "dropoutRate:0";


    public SavedModel(String importDir, String ModelTag) {
        SavedModelBundle sb = SavedModelBundle.load(importDir, ModelTag);
        this.s = sb.session();
    }

    /**
     * Function to predict the Number from a read File using the loaded NN Model
     *
     * @param inputArray transformed float Array, which represents the read File/Picture
     * @param estimator  boolean if it is an Estimator model, Estimator Model doesn't have dropout rate
     * @return the predicted (most likely to be) number as an integer
     */
    protected int predictNumber(float[] inputArray, boolean estimator) {
        Tensor result = getOutput(inputArray,estimator);
        return maxIndex(toArray(result));
    }

    protected Tensor getOutput(float[] inputArray, boolean estimator){
        Tensor inputTensor = toTensor(inputArray);
        Tensor dKeep = Tensor.create(1f);
        List<Tensor> resultList;
        if (!estimator) {
            resultList = this.s.runner().feed(input, inputTensor).feed(dropout, dKeep).fetch(tensorflowOutput).run();
        } else {
            resultList = this.s.runner().feed(input, inputTensor).fetch(estimatorOutput).run();
        }
        Tensor result = resultList.get(0);
        return result;
    }


}
