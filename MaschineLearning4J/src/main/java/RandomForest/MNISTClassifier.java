package RandomForest;

import java.io.*;
import java.util.*;

public class MNISTClassifier {
    private final static String importDir = "../Maschine Learning/RandomForest/MaaS/export/";
    private final static String n_estimators = "10";
    private final static String pmmlFile = importDir + "RandomForestMNIST_" + n_estimators + ".pmml";

    private final static String picDir = "../Maschine Learning/Data/Own_dat/"; //where the test pictures are stored
    private final static String picName = "MNIST-7.png"; //default

    private static String picPath = ""; //full path to picture
    private static boolean eval = true; //if results in java/python should be compared


    public static void main(String[] args) {
        evaluateArguments(args);

        System.out.println("Creating an evaluator from given PMML file: " + pmmlFile + ". \nDepending on the size of the RandomForest, this might take a while...");
        long time = System.nanoTime();
        RandomForestWrapper randomForest = new RandomForestWrapper(pmmlFile);
        long timeDifference = (System.nanoTime() - time) / 1000000;
        System.out.println("Finished creating the evaluator! Took " + timeDifference + "ms to finish.");
        //10->3,3s | 100->5,2s | 1000->40s

        if (eval) {
            String statisticsFile = importDir + "statistics_" + n_estimators + ".json";
            randomForest.compareResults(statisticsFile, picDir);
        }

        time = System.nanoTime();
        int prediction = randomForest.predict(picPath, false);
        timeDifference = (System.nanoTime() - time) / 1000000;
        System.out.println("The prediction call for given png, using the Random Forest, took " + timeDifference + "ms. (includes reading the pixel information)");
        //if only prediction: 10->1ms | 100->4ms |1000->17ms (loading is approximatly 160ms)
        System.out.println("--> The given picture at \"" + picPath + "\" is probably a: " + prediction);
    }

    /**
     * Evaluates given program arguments to determine from which png file to load its pixel(sets picPath) and wheter to compare Java/Python predictions
     * @param args program arguments passed forward from main function
     */
    private static void evaluateArguments(String[] args) {
        switch (args.length) {
            case 0:
                picPath = picDir + picName;
                System.out.println("Will use default picture to predict its number: " + picPath);
                break;
            case 1:
                if (args[0].equals("--noEval")) {
                    eval = false;
                    picPath = picDir + picName;
                    System.out.println("Will use default picture to predict its number: " + picPath);
                } else {
                    setPicPath(args[0]);
                }
                break;
            case 2:
                if (!Arrays.asList(args).contains("--noEval")) {
                    System.out.println("You have passed in invalid parameters, if you pass in 2 parameters, one has to be \"--noEval\".");
                    System.out.println("Please check given program arguments for typos: " + args[0] + " " + args[1]);
                    System.out.println("Exiting program...");
                    System.exit(1);
                } else {
                    String fileArgument = "";
                    if (args[0].equals("--noEval")) {
                        fileArgument = args[1];
                    } else {
                        fileArgument = args[0];
                    }
                    setPicPath(fileArgument);
                    eval = false;
                }
        }
    }
    /**
     * Utility function for evaluateArguments, sets "picPath", according to given argument. Given argument could be full path or filename in picdir, with or without .png ending
     * @param fileArgument given value of the program arguments
     */
    private static void setPicPath(String fileArgument) {
        String path = fileArgument;//assume full path given
        if (!path.contains(".png")) {
            path += ".png";
        }

        if (!new File(path).exists()) { //if no full path is given try picDir
            path = picDir + path;
        }

        File f = new File(path);
        if (f.exists() && !f.isDirectory()) {
            picPath = path;
            System.out.println("Will use given picture at: " + path + " to predict its number.");
        } else {
            System.out.println("The given file at: " + path + " is not a valid png file!");
            System.out.println("Exiting program...");
            System.exit(1);
        }
    }

}