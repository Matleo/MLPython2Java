package MNIST;

import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import java.util.List;
import static MNIST.TensorflowUtilities.*;


/**
 * class to load and store the trained Tensorflow Model
 */
public class SavedModel {
    private Session s;


    public SavedModel(String importDir, String ModelTag){
        SavedModelBundle sb= SavedModelBundle.load(importDir,ModelTag);
        this.s=sb.session();
    }

    /**
     * Function to predict the Number from a read File using the loaded NN Model
     * @param inputArray transformed float Array, which represents the read File/Picture
     * @param estimator boolean if it is an Estimator model, Estimator Model doesn't have dropout rate
     * @return the predicted (most likely to be) number as an integer
     */
    protected int predictNumber(float[] inputArray, boolean estimator){

        Tensor input = toTensor(inputArray);
        Tensor dKeep = Tensor.create(1f);
        List<Tensor> resultList;
        if(!estimator){
            resultList = this.s.runner().feed("input:0",input).feed("dropoutRate:0",dKeep).fetch("output:0").run();
        }else{
            resultList = this.s.runner().feed("input:0",input).fetch("dnn/head/predictions/probabilities:0").run();
        }
        Tensor result=resultList.get(0);

        return maxIndex(toArray(result));
    }


}
