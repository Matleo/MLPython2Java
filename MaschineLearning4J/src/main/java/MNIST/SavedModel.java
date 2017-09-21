package MNIST;


import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;
import org.tensorflow.Tensor;



import static MNIST.TFWrapper.*;


import java.util.List;



public class SavedModel {
    private Session s;
    private SavedModelBundle sb;


    public SavedModel(String importDir, String ModelTag){
        this.sb= SavedModelBundle.load(importDir,ModelTag);
        this.s=this.sb.session();
    }

    public int predictNumber(float[] inputArray){

        Tensor input = toTensor(inputArray);
        Tensor dKeep = Tensor.create(1f);
        List<Tensor> resultList = this.s.runner().feed("input:0",input).feed("dropoutRate:0",dKeep).fetch("output:0").run();
        Tensor result=resultList.get(0);

        return maxIndex(toArray(result));
    }


}
