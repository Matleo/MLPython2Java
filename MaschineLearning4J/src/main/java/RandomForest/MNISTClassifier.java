package RandomForest;

import NeuralNetwork.Tensorflow.MNIST.TensorflowUtilities;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.PMML;
import org.jpmml.evaluator.*;
import org.xml.sax.SAXException;

import javax.activation.MimetypesFileTypeMap;
import javax.xml.bind.JAXBException;
import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public class MNISTClassifier {
    private final static String file = "../Maschine Learning/RandomForest/MaaS/RandomForestMNIST.pmml";
    private final static String picDir = "../Maschine Learning/Data/Own_dat/"; //where the test pictures are stored
    private final static String picName = "MNIST-7.png"; //default
    private static String picPath = ""; //full path to picture


    public static void main(String[] args) {
        evaluateArguments(args);

        Evaluator evaluator = loadEvaluator();
        System.out.println("Created an evaluator from given PMML file: " + file);

        Map<FieldName, FieldValue> inputParameter = loadInputParameter(evaluator);
        System.out.println("Loaded pixel values from given picture into a float[] and loaded that array as input parameter into the model.");

        Map<FieldName, ?> results = evaluator.evaluate(inputParameter);
        System.out.println("Used the loaded pixel values to feed the model and get a prediction:");

        printResults(results, evaluator);
    }

    /**
     * Loads the Evaluator from defined PMML file ("file")
     *
     * @return Evaluator that encapsulates the loaded RandomForest Model from the PMML and can be used for prediction
     */
    private static Evaluator loadEvaluator() {
        InputStream is = null;
        try {
            is = new FileInputStream(file);
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
     * @return Map of FieldName->FieldValue, containing pixel values from the given png file
     */
    private static Map<FieldName, FieldValue> loadInputParameter(Evaluator evaluator) {
        float[] picture = TensorflowUtilities.readPic(picPath);

        Map<FieldName, FieldValue> arguments = new LinkedHashMap<>();
        List<InputField> inputFields = evaluator.getInputFields(); //{InputField{name=xi, dataType=FLOAT, opType=CONTINUOUS}}; i in int(0,784); if feature_importance of that pixel is 0, the feature is not included. so inputFields.length<784
        for (int i = 0; i < inputFields.size(); i++) {
            FieldName inputFieldName = inputFields.get(i).getName();//xi with i in int(0,784)
            int index = Integer.valueOf(inputFieldName.toString().substring(1)); //i in int(0,784)

            Object rawValue = picture[index];
            FieldValue inputFieldValue = inputFields.get(i).prepare(rawValue);
            arguments.put(inputFieldName, inputFieldValue);
        }
        return arguments;
    }

    /**
     * Prints the results that were calculated from the evaluator to the console
     *
     * @param results   return value of the evaluator.evaluate() function
     * @param evaluator Evaluator that was created by the loadEvaluator() function, from the defined PMML file
     */
    private static void printResults(Map<FieldName, ?> results, Evaluator evaluator) {
        List<TargetField> targetFields = evaluator.getTargetFields();
        for (TargetField targetField : targetFields) {
            FieldName targetFieldName = targetField.getName();

            //ProbabilityDistribution{result=7.0, probability_entries=[0.0=0.0, 1.0=0.0, 2.0=0.0, 3.0=0.0, 4.0=0.0, 5.0=0.0, 6.0=0.0, 7.0=1.0, 8.0=0.0, 9.0=0.0]}
            ProbabilityDistribution targetFieldValue = (ProbabilityDistribution) results.get(targetFieldName);
            int prediction = ((Double) targetFieldValue.getResult()).intValue();
            System.out.println("\n--> The given picture at \"" + picPath + "\" is probably a: " + prediction);

        }
    }

    /**
     * Evaluates given program arguments to determine from which png file to load its pixel(sets picPath) and use that to predict its displayed number
     *
     * @param args program arguments passed forward from main function
     */
    private static void evaluateArguments(String[] args) {
        switch (args.length) {
            case 0:
                picPath=picDir+picName;
                System.out.println("Using default picture for predicting: "+picPath);
                break;
            case 1:
                String path = args[0];//assume full path given
                if (!path.contains(".png")) {
                    path += ".png";
                }

                if (!new File(path).exists()) { //if no full path is given try picDir
                    path = picDir + path;
                }

                File f = new File(path);
                if (f.exists() && !f.isDirectory()) {
                    picPath = path;
                    System.out.println("Using given picture at: " + path + " for predicting.");
                } else {
                    System.out.println("The given file at: " + path + " is not a valid png file!");
                    System.out.println("Exiting program...");
                    System.exit(1);
                }
        }
    }
}