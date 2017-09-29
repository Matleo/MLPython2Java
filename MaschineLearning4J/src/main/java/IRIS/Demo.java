package IRIS;

import java.io.File;
import java.io.FileFilter;
import java.nio.FloatBuffer;
import java.util.List;

import MNIST.TensorflowUtilities;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

/**
 * simple class with Tensorflow API to load a pre-trained tensorflow.Estimator model and test it against the IRIS data set
 */
public class Demo {
    public static void main(String[] args) {
        final String importdir = "C:/Users/lema/IdeaProjects/Maschine Learning/NeuralNetwork/Estimator/IrisClassifier/export/";
        String timestamp = getTimestampDir(importdir);//because the Estimator stores the model in a new directory with a timestamp

        Session session = SavedModelBundle.load(timestamp, "serve").session();

        //test Tensor
        Tensor x = Tensor.create( new long[] {2,4}, FloatBuffer.wrap(
                new float[] {   6.4f, 3.2f, 4.5f, 1.5f,
                                6.5f,3.0f,5.2f,2.0f}));

        //names of Tensors in the model are hardcoded for the moment
        final String inputName = "input:0";
        final String scoresName = "dnn/head/predictions/probabilities:0";

        List<Tensor> outputs = session.runner().feed(inputName, x).fetch(scoresName).run();

        // Outer dimension is batch size; inner dimension is number of classes
        float[][] scores = new float[2][3];
        outputs.get(0).copyTo(scores);//copy output Tensor to scores Array

        for (int i=0; i<scores.length;i++){
            int predictionNumber = TensorflowUtilities.maxIndex(scores[i]);
            switch(predictionNumber){
                case 0:
                    System.out.println("Class Prediction of individual "+i+": Iris setosa");
                    break;
                case 1:
                    System.out.println("Class Prediction of individual "+i+": Iris versicolor");
                    break;
                case 2:
                    System.out.println("Class Prediction of individual "+i+": Iris virginica");
                    break;
            }
        }
    }

    //Reads first (and only) sub directory/timestamp in export folder
    public static String getTimestampDir(String directory){
        File dir = new File(directory);
        File[] subDirectories = dir.listFiles(new FileFilter() {
            public boolean accept(File file) {
                return file.isDirectory();
            }
        });
        return subDirectories[0].toString();
    }
}
