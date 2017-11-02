package RandomForest;

import NeuralNetwork.Tensorflow.MNIST.TensorflowUtilities;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.PMML;
import org.jpmml.evaluator.*;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;
import org.xml.sax.SAXException;

import javax.activation.MimetypesFileTypeMap;
import javax.xml.bind.JAXBException;
import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

public class MNISTClassifier {
    private final static String importDir = "../Maschine Learning/RandomForest/MaaS/";
    private final static String pmmlFile = importDir + "RandomForestMNIST.pmml";
    private final static String picDir = "../Maschine Learning/Data/Own_dat/"; //where the test pictures are stored
    private final static String picName = "MNIST-7.json"; //default

    private static String picPath = ""; //full path to picture
    private static boolean eval = true; //if results in java/python should be compared


    public static void main(String[] args) {
        evaluateArguments(args);

        System.out.println("Creating an evaluator from given PMML file: " + pmmlFile + ". Depending on the size of the RandomForest, this might take a while...");
        Evaluator evaluator = loadEvaluator();
        System.out.println("Finished creating the evaluator!");

        if (eval) compareResults(evaluator);

        Map<FieldName, FieldValue> inputParameter = loadInputParameter(evaluator, picPath, false);

        Map<FieldName, ?> results = evaluator.evaluate(inputParameter);

        int prediction = readResult(evaluator, results);
        System.out.println("\n--> The given picture at \"" + picPath + "\" is probably a: " + prediction);
    }

    /**
     * Evaluates given program arguments to determine from which png file to load its pixel(sets picPath) and use that to predict its displayed number
     *
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
     *
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

    /**
     * Loads the Evaluator from defined PMML file ("file")
     *
     * @return Evaluator that encapsulates the loaded RandomForest Model from the PMML and can be used for prediction
     */
    private static Evaluator loadEvaluator() {
        InputStream is = null;
        try {
            is = new FileInputStream(pmmlFile);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }

        PMML pmml = null;
        try {
            pmml = org.jpmml.model.PMMLUtil.unmarshal(is);
        } catch (SAXException e) {
            e.printStackTrace();
        } catch (JAXBException e) {
            e.printStackTrace();
        }
        ModelEvaluatorFactory modelEvaluatorFactory = ModelEvaluatorFactory.newInstance();

        ModelEvaluator<?> modelEvaluator = modelEvaluatorFactory.newModelEvaluator(pmml);
        return modelEvaluator;
    }

    /**
     * Loads the pixel information from the given picture at picPath into a Map that can be passed to the evaluator
     *
     * @param evaluator Evaluator that was created from the PMML file
     * @param pic       String of the full path to the file that should be loaded
     * @return Map of FieldName->FieldValue, containing pixel values from the given png file
     */
    private static Map<FieldName, FieldValue> loadInputParameter(Evaluator evaluator, String pic, boolean loadJson) {
        float[] picture;
        if (loadJson) {
            picture = TensorflowUtilities.readJsonPic(pic);
        } else {
            picture = TensorflowUtilities.readPic(pic);
        }

        Map<FieldName, FieldValue> arguments = new LinkedHashMap<>();
        List<InputField> inputFields = evaluator.getInputFields(); //{InputField{name=xi, dataType=FLOAT, opType=CONTINUOUS}}; i in int(1,784); if feature_importance of that pixel is 0, the feature is not included. so inputFields.length<784

        for (int i = 0; i < inputFields.size(); i++) {
            FieldName inputFieldName = inputFields.get(i).getName();//xi with i in int(1,784)
            int index = Integer.valueOf(inputFieldName.toString().substring(1)); //i in int(1,784)

            Object rawValue = picture[index - 1]; //-1 because indices of DataFields are in int(1,784) and x1 -> first pixel -> picture[0]
            FieldValue inputFieldValue = inputFields.get(i).prepare(rawValue);
            arguments.put(inputFieldName, inputFieldValue);
        }

        return arguments;
    }


    private static boolean compareResults(Evaluator evaluator) {
        JSONParser parser = new JSONParser();
        Map<String, int[]> picPredictionsJ = new HashMap<>();
        Map<String, int[]> picPredictionsP = new HashMap<>();

        try {
            //read out statistics.json:
            JSONObject obj = (JSONObject) parser.parse(new FileReader(importDir + "statistics.json"));
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

                int[] predictionsJ = getPicCatPredictions(evaluator, picCat);//make java predictions

                picPredictionsP.put(picCat, predictionsP);
                picPredictionsJ.put(picCat, predictionsJ);
            }


        } catch (ParseException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }

        boolean match = TensorflowUtilities.compareMaps(picPredictionsJ, picPredictionsP);
        return match;
    }

    private static int[] getPicCatPredictions(Evaluator evaluator, String picCat) {
        int[] predictions = new int[10];
        for (int i = 0; i < 10; i++) {
            String path = picDir + picCat + "-" + i + ".json";
            Map<FieldName, FieldValue> inputParameter = loadInputParameter(evaluator, path, true);
            Map<FieldName, ?> results = evaluator.evaluate(inputParameter);
            int prediction = readResult(evaluator, results);
            predictions[i] = prediction;
        }
        return predictions;
    }

    /**
     * Reads out the result object
     *
     * @param results   return value of the evaluator.evaluate() function
     * @param evaluator Evaluator that was created by the loadEvaluator() function, from the defined PMML file
     * @return integer representing the predicted number, which is displayed on the given image
     */
    private static int readResult(Evaluator evaluator, Map<FieldName, ?> results) {
        List<TargetField> targetFields = evaluator.getTargetFields();
        TargetField targetField = targetFields.get(0);
        FieldName targetFieldName = targetField.getName();

        //ProbabilityDistribution{result=7.0, probability_entries=[0.0=0.0, 1.0=0.0, 2.0=0.0, 3.0=0.0, 4.0=0.0, 5.0=0.0, 6.0=0.0, 7.0=1.0, 8.0=0.0, 9.0=0.0]}
        ProbabilityDistribution targetFieldValue = (ProbabilityDistribution) results.get(targetFieldName);
        int prediction = ((Double) targetFieldValue.getResult()).intValue();
        return prediction;
    }

    private static void printUnusedPixels(Map<FieldName, FieldValue> inputParameter, String pic) {
        List<Integer> presentFeatures = new LinkedList<>(); //all indices of present DataFields
        for (FieldName fn : inputParameter.keySet()) {
            int index = Integer.valueOf(fn.toString().substring(1)); //i in int(1,784)
            presentFeatures.add(index);
        }

        float[] picture = TensorflowUtilities.readJsonPic(pic);
        Set<Integer> unpresentFeatures = new HashSet<>();
        for (int i = 1; i <= picture.length; i++) {
            unpresentFeatures.add(i);//[1-784]
        }
        for (int i : presentFeatures) {
            unpresentFeatures.remove(i);//remove all present indices
        }

        for (int i : unpresentFeatures) {
            System.out.println(i + ": " + picture[i - 1]);
        }
    }
}