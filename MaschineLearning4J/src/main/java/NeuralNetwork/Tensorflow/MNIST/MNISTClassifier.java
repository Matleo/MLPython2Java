package NeuralNetwork.Tensorflow.MNIST;

import java.io.File;
import java.io.FileFilter;
import java.util.Arrays;
import java.util.List;

/**
 * Main Class to load a SavedModel from Tensorflow or Tensorflow.Estimator or Keras with Tensorflow backend,
 * to infer the MNSIT dataset and test the model against additionaldownloaded pictures
 * pass program arguments:
 * -m | -model <value>     specify which SavedModel to use for predictions
 * --noEval                pass if the imported model should NOT be evaluated against the saved results from the python model
 * -p | -predict <path>    specify of which picture to predict its number
 */
public class MNISTClassifier {
    private static final String picDir = "../Maschine Learning/Data/Own_dat/"; //where the test pictures are stored
    private static final String importDir1 = "../Maschine Learning/NeuralNetwork/Estimator/MNISTClassifier/FFNN/export/";
    private static final String importDir2 = "../Maschine Learning/NeuralNetwork/Tensorflow/MNISTClassifier/CNN/export/";
    private static final String importDir3 = "../Maschine Learning/NeuralNetwork/Tensorflow/MNISTClassifier/Feed Forward NN/SavedModel/export/";
    private static final String importDir4 = "../Maschine Learning/NeuralNetwork/Estimator/MNISTClassifier/CNN/export/";
    private static final String importDir5 = "../Maschine Learning/NeuralNetwork/Keras/MNISTClassifier/Model/export/";
    private static final String importDir6 = "../Maschine Learning/NeuralNetwork/Keras/MNISTClassifier/Sequential/export/";
    private static final String modelTag = "serve"; //default Tag under which the Metagraph is stored in the SavedModel

    private static String importDir = "";//which model to use, will be filled by the evaluateArguments() function
    private static boolean eval = true; //if results in java/python should be compared
    private static String predictPicFile = "";//will maybe be filled by the evaluateArguments() function. What picture to predict number of
    private static String modelType = ""; //define what kind of model is going to be loaded, will be filled by evaluateArguments() function. "Tensorflow" is default


    public static void main(String[] args) throws Exception {
        System.out.print("Program Arguments:");
        for (String a : args) {
            System.out.print(" " + a);
        }
        evaluateArguments(args);

        SavedModel model = new SavedModel(importDir, modelTag, modelType);
        System.out.println("\nLoaded model from: " + importDir);


        if (eval) model.evaluate(importDir,picDir);

        if (!predictPicFile.equals("")) {
            int prediction = model.predictImage(predictPicFile);
            System.out.println("\nThe given picture at " + predictPicFile + " is probably a: " + prediction);
        }
    }


    /**
     * Assign correct values to importdir, predictPicFile and modelType, according to input program arguments
     * @param args program arguments forwarded in the main method
     */
    private static void evaluateArguments(String[] args) {
        System.out.println();
        switch (args.length) {
            case 0:
                System.out.println("Using default model: Tensorflow_FFNN");
                modelType = "Tensorflow";
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
                    predictPicFile = args[1];
                    System.out.println("Using default model: Tensorflow_FFNN");
                    modelType = "Tensorflow";
                    importDir = importDir3;//default Tensorflow_FFNN
                } else {
                    System.out.println("You have passed invalid arguments");
                    printHelp();
                    System.exit(1);
                }
                break;
            case 3:
                System.out.println("Using default model: Tensorflow_FFNN");
                modelType = "Tensorflow";
                importDir = importDir3;//default Tensorflow_FFNN
                if (((args[0].equals("-predict") || args[0].equals("-p")) && (args[2].equals("--noEval")))) {
                    predictPicFile = args[1];
                    eval = false;
                } else if ((args[1].equals("-predict") || args[1].equals("-p")) && (args[0].equals("--noEval"))) {
                    predictPicFile = args[2];
                    eval = false;
                } else {
                    System.out.println("You have passed invalid arguments");
                    printHelp();
                    System.exit(1);
                }
                break;
            case 4:
                if ((args[0].equals("-predict") || args[0].equals("-p")) && (args[2].equals("-model") || args[2].equals("-m"))) {
                    predictPicFile = args[1];
                    importDir = determinModel(args[3]);
                } else if ((args[2].equals("-predict") || args[2].equals("-p")) && (args[0].equals("-model") || args[0].equals("-m"))) {
                    predictPicFile = args[3];
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
                    predictPicFile = args[indexP + 1];
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

    /**
     * Utility function to evaluate program argument modelType
     *
     * @param model input String for the model parameter
     * @return String of according importDir, of where to import the ML model
     */
    private static String determinModel(String model) {
        switch (model) {
            case "t_ffnn":
                modelType = "Tensorflow";
                return importDir3;
            case "t_cnn":
                modelType = "Tensorflow";
                return importDir2;
            case "e_ffnn":
                modelType = "Estimator";
                //get the only subdirectory of the export directory, which is the timestamp, generated by Tensorflow.Estimator while saving
                File dir = new File(importDir1);
                File[] subDirectories = dir.listFiles(new FileFilter() {
                    public boolean accept(File file) {
                        return file.isDirectory();
                    }
                });
                return subDirectories[0].toString();
            case "e_cnn":
                modelType = "Estimator";
                //get the only subdirectory of the export directory, which is the timestamp, generated by Tensorflow.Estimator while saving
                File dir2 = new File(importDir4);
                File[] subDirectories2 = dir2.listFiles(new FileFilter() {
                    public boolean accept(File file) {
                        return file.isDirectory();
                    }
                });
                return subDirectories2[0].toString();
            case "k_cnn":
                modelType = "KerasModel";
                return importDir5;
            case "k_ffnn":
                modelType = "KerasSequential";
                return importDir6;
            default:
                return "";
        }
    }


    private static void printHelp() {
        System.out.println("Valid arguments include: ");
        System.out.println("    -m | -model <value>     specify which SavedModel to use for predictions");
        System.out.println("                            value can be any of \"t_ffnn\" / \"t_cnn\" / \"e_ffnn\"/ \"e_cnn\" / \"k_ffnn\"/ \"k_cnn\"");
        System.out.println("                            if you don't pass this argument, default Tensorflow_FeedForwardNeuralNet will be used");
        System.out.println("    --noEval                pass if the imported model should NOT be evaluated against the saved results from the python model");
        System.out.println("                            if you pass this argument, you need to specify -p");
        System.out.println("    -p | -predict <path>    specify of which picture to predict its number");
        System.out.println("                            needs to be a absolute path and pointing to a .png file");
        System.out.println("                            if you don't pass this argument, no prediction will be executed");

    }





}