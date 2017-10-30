package NeuralNetwork.Tensorflow.MNIST;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.tensorflow.*;

import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * class to load, store and use a previously saved SavedModel
 */
public class SavedModel {
    private Session session;
    private String modelType;

    private static final String input = "input:0";
    private static final String output = "output:0";
    private static final String dropout = "dropoutRate:0";
    private static final String kerasInput = "input_input:0";
    private static final String kerasOutput = "output/Softmax:0";
    private static final String kerasLearningPhase = "dropout_1/keras_learning_phase:0";


    public SavedModel(String importDir, String modelTag, String modelType) {
        SavedModelBundle sb = SavedModelBundle.load(importDir, modelTag);
        this.session = sb.session();
        this.modelType = modelType;
    }

    /**
     * Function to predict the Number from a float[]
     * @param inputArray transformed float Array, which represents the read File/Picture
     * @return the predicted (most likely to be) number as an integer
     */
    protected int predict(float[] inputArray) {
        Tensor result = getOutput(inputArray);
        return TensorflowUtilities.maxIndex(TensorflowUtilities.toArray(result));
    }

    /**
     * Fetches the output Tensor from the model, depending on the modelType
     * @param inputArray transformed float Array, which represents the read File/Picture
     * @return Tensor containing the probabilities/scores for the prediction
     */
    protected Tensor getOutput(float[] inputArray) {
        Tensor inputTensor = TensorflowUtilities.toTensor(inputArray);

        List<Tensor> resultList;
        if (modelType == "Tensorflow") {
            Tensor dKeep = Tensor.create(1f);
            resultList = this.session.runner().feed(input, inputTensor).feed(dropout, dKeep).fetch(output).run();
        } else if (modelType == "Estimator") {
            resultList = this.session.runner().feed(input, inputTensor).fetch(output).run();
        } else {//KerasModel or KerasSequential
            Tensor learningPhase = Tensor.create(false);
            resultList = this.session.runner().feed(kerasInput, inputTensor).feed(kerasLearningPhase, learningPhase).fetch(kerasOutput).run();
        }

        Tensor result = resultList.get(0);
        return result;
    }

    /**
     * Predicts a number from a given image(file)
     * @param pathfile file of where to find the image to predict
     * @return predicted number as integer
     */
    protected int predictImage(String pathfile) {
        if (!pathfile.contains(".png")) pathfile = pathfile + ".png";
        float[] inputArray = TensorflowUtilities.readPic(pathfile);
        int predict = predict(inputArray);
        return predict;
    }

    /**
     * Predict a batch of saved images. Used to validate against Python results
     * @param picCategory name of the picture category, any of "MNIST"/"Font"/"Computer"/"Handwritten"
     * @return array of size 10, containing predictions for images 0-9, at corresponding index (array[3] is the prediction for the image of a 3)
     */
    protected int[] getPredictions(String picCategory, String picDir) {
        int[] predictions = new int[10];
        for (int i = 0; i < 10; i++) {
            float[] inputArray = TensorflowUtilities.readJsonPic(picDir + picCategory + "-" + i + ".json");
            int predict = predict(inputArray);
            predictions[i] = predict;
        }
        return predictions;
    }

    /**
     * Evaluates the Java predictions against the mnist data set and previously saved results from the Python predictions
     * @param importDir Directory of where to find the statistics.json
     * @param picDir Directory of where to find the pics to make predictions of
     * @return wether the Java predictions match the Python predictions
     * @throws IOException when can not open the statistics.json file
     */
    protected boolean evaluate(String importDir, String picDir) throws IOException {

        System.out.println("\nEvaluating against the MNSIT Dataset...");
        Evaluation evaluator = new Evaluation(10); //create an evaluation object with 10 possible classes
        DataSetIterator mnistTest = new MnistDataSetIterator(1, false, 123);

        while (mnistTest.hasNext()) {
            DataSet next = mnistTest.next(); //(1,784)
            INDArray next2 = next.getFeatureMatrix();//(1,784)
            float[] array = new float[next2.shape()[1]];
            //convert INDArray to float[]:
            for (int i = 0; i < next2.shape()[1]; i++) {
                array[i] = next2.getFloat(i);
            }
            Tensor output = getOutput(array);
            float[] outputArray = TensorflowUtilities.toArray(output);

            INDArray outputNDArray = Nd4j.create(outputArray);
            evaluator.eval(next.getLabels(), outputNDArray); //check the prediction against the true class
        }


        double javAccuracy = evaluator.accuracy();
        double pyAccuracy = -1;
        JSONParser parser = new JSONParser();
        Map<String, int[]> picPredictionsJ = new HashMap<>();
        Map<String, int[]> picPredictionsP = new HashMap<>();

        try {
            //read out statistics.json:
            JSONObject obj = (JSONObject) parser.parse(new FileReader(importDir + "/statistics.json"));
            pyAccuracy = (double) obj.get("accuracy");
            JSONObject picPredictionsJson = (JSONObject) obj.get("picPredictions");

            //make own prediction on pictures
            Object[] picCats = picPredictionsJson.keySet().toArray(); //names of all picture category (MNIST,Font,Computer,Handwritten)
            for (Object picCatObj : picCats) {
                String picCat = (String) picCatObj;//name of the picture category
                JSONArray jsonPredictionsP = (JSONArray) picPredictionsJson.get(picCat);
                int[] predictionsP = new int[10];//python predictions
                for (int i = 0; i < jsonPredictionsP.size(); i++) {
                    predictionsP[i] = (int) (long) (jsonPredictionsP.get(i));
                }

                int[] predictionsJ = getPredictions(picCat, picDir);//make java predictions

                picPredictionsP.put(picCat, predictionsP);
                picPredictionsJ.put(picCat, predictionsJ);
            }


        } catch (ParseException e) {
            e.printStackTrace();
        }

        System.out.println(evaluator.stats());
        boolean match = (javAccuracy == pyAccuracy);
        if (match) {
            System.out.println("\n***Success***");
            System.out.println("The calculated accuracy on the MNIST dataset in Java and Python match");
        } else {
            System.out.println("\nSomething went wrong, the accuracy calculated in java and python don't match");
            System.out.println("Java accuracy: " + javAccuracy);
            System.out.println("Python accuracy: " + pyAccuracy);
        }

        boolean picPredictionsMatch = TensorflowUtilities.compareMaps(picPredictionsJ, picPredictionsP);

        if (match && picPredictionsMatch) {
            return true;
        } else {
            return false;
        }
    }
}
