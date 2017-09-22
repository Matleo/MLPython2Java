package IRIS;

import java.nio.FloatBuffer;
import java.util.List;

import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

public class Demo {
    public static void main(String[] args) {

        final String importdir = "C:/Users/lema/IdeaProjects/Maschine Learning/NeuralNetwork/Estimator/IrisClassifier/export/";
        String timestamp = "1506098146";

        Session session = SavedModelBundle.load(importdir+timestamp, "serve").session();

        Tensor x = Tensor.create( new long[] {2,4}, FloatBuffer.wrap(
                new float[] {   6.4f, 3.2f, 4.5f, 1.5f,
                                6.5f,3.0f,5.2f,2.0f}));


        final String inputName = "input:0";
        final String scoresName = "dnn/head/predictions/probabilities:0";

        List<Tensor> outputs = session.runner().feed(inputName, x).fetch(scoresName).run();

        // Outer dimension is batch size; inner dimension is number of classes
        float[][] scores = new float[2][3];
        outputs.get(0).copyTo(scores);

        for (int i=0; i<scores.length;i++){
            int predictionNumber = MNIST.TFWrapper.maxIndex(scores[i]);
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
}
