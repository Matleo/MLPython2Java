package Tensorflow;

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
import org.tensorflow.Tensor;

import java.io.File;
import java.io.FileFilter;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Execution Class to load a SavedModel from Tensorflow or Tensorflow.Estimator to recognize the MNSIT data Set, and test it against downloaded pictures
 */
public class MNISTClassifier {
    private static final String picDir = "C:/Users/lema/IdeaProjects/Maschine Learning/NeuralNetwork/Data/Own_dat/"; //where the test pics are stored
    private static String importDir1 = "../Maschine Learning/NeuralNetwork/Estimator/MNISTClassifier/export/";
    private static String importDir2 = "../Maschine Learning/NeuralNetwork/Tensorflow/CNN/export/";
    private static String importDir3 = "../Maschine Learning/NeuralNetwork/Tensorflow/Feed Forward NN/SavedModel/export/";

    private static String importDir = "";//which model to use
    private static boolean eval = true; //if results in java/python should be compared
    private static String predictPicDir = "";//if not "", will be used to use trained model to predict the picture

    private static final String modelTag = "serve"; //default Tag under which the Metagraph is stored in the SavedModel
    private static boolean estimator = false; //if it is an estimator model


    public static void main(String[] args) throws Exception {
        System.out.print("Program Arguments:");
        for (String a : args) {
            System.out.print(" " + a);
        }

        evaluateArguments(args);

        System.out.println("\nLoaded model from: " + importDir);
        SavedModel model = new SavedModel(importDir, modelTag);


        if (eval) evaluate(model, estimator);

        if (!predictPicDir.equals("")) {
            predict(model, predictPicDir);
        }
    }


    //assign correct values to importdir and picFile, according to input program arguments
    private static void evaluateArguments(String[] args) {
        System.out.println();
        switch (args.length) {
            case 0:
                System.out.println("Using default model: Tensorflow_FFNN");
                importDir = importDir3;//default Tensorflow_FFNN
                break;
            case 1:
                if (args[0].contains("-h")) {
                    printHelp();
                    System.exit(0);
                } else {
                    System.out.println("You passed an invalid amount of arguments");
                    printHelp();
                    System.exit(1);
                }
            case 2:
                if (args[0].equals("-model") || args[0].equals("-m")) {
                    importDir = determinModel(args[1]);
                } else if (args[0].equals("-predict") || args[0].equals("-p")) {
                    predictPicDir = args[1];
                    System.out.println("Using default model: Tensorflow_FFNN");
                    importDir = importDir3;//default Tensorflow_FFNN
                } else {
                    System.out.println("You have passed invalid arguments");
                    printHelp();
                    System.exit(1);
                }
                break;
            case 3:
                System.out.println("Using default model: Tensorflow_FFNN");
                importDir = importDir3;//default Tensorflow_FFNN
                if (((args[0].equals("-predict") || args[0].equals("-p")) && (args[2].equals("--noEval")))) {
                    predictPicDir = args[1];
                    eval = false;
                } else if ((args[1].equals("-predict") || args[1].equals("-p")) && (args[0].equals("--noEval"))) {
                    predictPicDir = args[2];
                    eval = false;
                } else {
                    System.out.println("You have passed invalid arguments");
                    printHelp();
                    System.exit(1);
                }
                break;
            case 4:
                if ((args[0].equals("-predict") || args[0].equals("-p")) && (args[2].equals("-model") || args[2].equals("-m"))) {
                    predictPicDir = args[1];
                    importDir = determinModel(args[3]);
                } else if ((args[2].equals("-predict") || args[2].equals("-p")) && (args[0].equals("-model") || args[0].equals("-m"))) {
                    predictPicDir = args[3];
                    importDir = determinModel(args[1]);
                } else {
                    System.out.println("You have passed invalid arguments");
                    printHelp();
                    System.exit(1);
                }
                break;
            case 5:
                List argsList = Arrays.asList(args);
                boolean containsNE = argsList.contains("--noEval");
                boolean containsM1 = argsList.contains("-m");
                boolean containsM2 = argsList.contains("-model");
                boolean containsP1 = argsList.contains("-p");
                boolean containsP2 = argsList.contains("-predict");
                if (containsNE && (containsM1 || containsM2) && (containsP1 || containsP2)) {
                    int indexM = -1;
                    if (containsM1) {
                        indexM = argsList.indexOf("-m");
                    } else {
                        indexM = argsList.indexOf("-model");
                    }
                    int indexP = -1;
                    if (containsP1) {
                        indexP = argsList.indexOf("-p");
                    } else {
                        indexP = argsList.indexOf("-predict");
                    }


                    eval = false;
                    predictPicDir = args[indexP + 1];
                    importDir = determinModel(args[indexM + 1]);
                } else {
                    System.out.println("You have passed invalid arguments");
                    printHelp();
                    System.exit(1);
                }
                break;
            default:
                System.out.println("You passed an invalid amount of arguments");
                printHelp();
                System.exit(1);
        }
    }

    private static String determinModel(String model) {
        switch (model) {
            case "t_ffnn":
                return importDir3;
            case "t_cnn":
                return importDir2;
            case "e_dnn":
                estimator = true;//pass for fetching output-> no explicit dropout needs to be fed

                //get the only subdirectory of the export directory, which is the timestamp, generated by Tensorflow.Estimator while saving
                File dir = new File(importDir1);
                File[] subDirectories = dir.listFiles(new FileFilter() {
                    public boolean accept(File file) {
                        return file.isDirectory();
                    }
                });
                return subDirectories[0].toString();
            default:
                return "";
        }
    }


    private static void printHelp() {
        System.out.println("Valid arguments include: ");
        System.out.println("    -m | -model <value>     specify which SavedModel to use for predictions");
        System.out.println("                            value can be any of \"t_ffnn\" / \"t_cnn\" / \"e_dnn\"");
        System.out.println("                            if you don't pass this argument, default Tensorflow_FeedForwardNeuralNet will be used");
        System.out.println("    --noEval                pass if the imported model should NOT be evaluated against the saved results from the python model");
        System.out.println("                            if you pass this argument, you need to specify -p");
        System.out.println("    -p | -predict <path>    specify of which picture to predict its number");
        System.out.println("                            path can be relative to this project or absolute and pointing to a .png File");
        System.out.println("                            if you don't pass this argument, no prediction will be executed");

    }

    //evaluates the predictions against the mnist data set -> returns accuracy
    //accuracies match for t_ffnn and t_cnn
    private static void evaluate(SavedModel model, boolean estimator) throws IOException {

        System.out.println("\nEvaluating against the MNSIT Dataset...");
        Evaluation evaluator = new Evaluation(10); //create an evaluation object with 10 possible classes
        DataSetIterator mnistTest = new MnistDataSetIterator(1, false, 123);

        while (mnistTest.hasNext()) {
            DataSet next = mnistTest.next(); //(1,784)
            INDArray next2 = next.getFeatureMatrix();//(1,784)
            float[] array = new float[next2.shape()[1]];
            for (int i = 0; i < next2.shape()[1]; i++) {
                array[i] = next2.getFloat(i);
            }
            Tensor output = model.getOutput(array, estimator);
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
            JSONObject obj = (JSONObject) parser.parse(new FileReader(importDir + "/statistics.json"));
            pyAccuracy = (double) obj.get("accuracy");
            JSONObject picPredictionsJson = (JSONObject) obj.get("picPredictions");

            //make own prediction on pictures

            Object[] picCats = picPredictionsJson.keySet().toArray();
            for (Object picCatObj : picCats) {
                String picCat = (String) picCatObj;//name of the picture category
                JSONArray jsonPredictionsP = (JSONArray) picPredictionsJson.get(picCat);
                int[] predictionsP = new int[10];//python predictions
                for (int i = 0; i < jsonPredictionsP.size(); i++) {
                    predictionsP[i] = (int) (long) (jsonPredictionsP.get(i));
                }

                int[] predictionsJ = getPredictions(model, picCat);//java predictions

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
            System.out.println("The calculated accuracy on the MNIST dataset in java and python match");
        } else {
            System.out.println("\nSomething went wrong, the accuracy calculated in java and python don't match");
            System.out.println("Java accuracy: " + javAccuracy);
            System.out.println("Python accuracy: " + pyAccuracy);
        }

        TensorflowUtilities.compareMaps(picPredictionsJ, picPredictionsP);

    }

    private static void predict(SavedModel model, String pathfile) {
        if (!pathfile.contains(".png")) pathfile = pathfile + ".png";
        float[] inputArray = TensorflowUtilities.readPic(pathfile);
        int predict = model.predictNumber(inputArray, estimator);
        System.out.println("\nThe given picture at " + pathfile + " is probably a: " + predict);
    }

    private static int[] getPredictions(SavedModel model, String pics) {
        int[] predictions = new int[10];
        for (int i = 0; i < 10; i++) {
            float[] inputArray = TensorflowUtilities.readJsonPic(picDir + pics + "-" + i + ".json");
            int predict = model.predictNumber(inputArray, estimator);
            predictions[i] = predict;
        }
        return predictions;
    }
}