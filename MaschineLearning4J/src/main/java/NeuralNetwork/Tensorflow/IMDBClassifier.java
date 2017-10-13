package NeuralNetwork.Tensorflow;

import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import java.nio.FloatBuffer;
import java.util.List;

public class IMDBClassifier {
    private static final String importDir = "../Maschine Learning/NeuralNetwork/Keras/IMDBClassifier/export";
    private static final String modelTag = "serve";
    private static final String input = "input_1:0";
    private static final String output = "dense_1/Sigmoid:0";
    private static float[] posExample = {15, 256, 4, 2, 7, 3766, 5, 723, 36, 71, 43, 530
            , 476, 26, 400, 317, 46, 7, 4, 12118, 1029, 13, 104, 88
            , 4, 381, 15, 297, 98, 32, 2071, 56, 26, 141, 6, 194
            , 7486, 18, 4, 226, 22, 21, 134, 476, 26, 480, 5, 144
            , 30, 5535, 18, 51, 36, 28, 224, 92, 25, 104, 4, 226
            , 65, 16, 38, 1334, 88, 12, 16, 283, 5, 16, 4472, 113
            , 103, 32, 15, 16, 5345, 19, 178, 32};
    private static float[] negExample = {125, 68, 2, 6853, 15, 349, 165, 4362, 98, 5, 4, 228, 9, 43, 2,
            1157, 15, 299, 120, 5, 120, 174, 11, 220, 175, 136, 50, 9, 4373, 228,
            8255, 5, 2, 656, 245, 2350, 5, 4, 9837, 131, 152, 491, 18, 2, 32,
            7464, 1212, 14, 9, 6, 371, 78, 22, 625, 64, 1382, 9, 8, 168, 145,
            23, 4, 1690, 15, 16, 4, 1355, 5, 28, 6, 52, 154, 462, 33, 89,
            78, 285, 16, 145, 95};


    private static Session session;

    public static void main(String[] args) {
        SavedModelBundle sb = SavedModelBundle.load(importDir, modelTag);
        session = sb.session();

        float negPrediction = predict(negExample);
        float posPrediction = predict(posExample);

        System.out.println("Info: The predictions will be in (0,1). The higher the value is, the higher the probability of it being a good review");
        System.out.println("Predictions for positive example: " + posPrediction);
        System.out.println("Predictions for negative example: " + negPrediction);
    }

    private static float predict(float[] inputExample) {
        float[][] example = new float[1][];
        example[0] = inputExample;

        Tensor inputTensor = Tensor.create(example);
        List<Tensor> resultList = session.runner().feed(input, inputTensor).fetch(output).run();
        Tensor resultTensor = resultList.get(0);

        FloatBuffer fb = FloatBuffer.allocate(resultTensor.numElements());
        resultTensor.writeTo(fb);

        return fb.get(0);
    }
}
