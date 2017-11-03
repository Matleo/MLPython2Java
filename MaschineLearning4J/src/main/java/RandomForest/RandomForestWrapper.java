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

import javax.xml.bind.JAXBException;
import java.io.*;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public class RandomForestWrapper {
    private ModelEvaluator<?> evaluator;

    /**
     * Loads the Evaluator from defined PMML file and stores it as object attribute
     *
     * @param pmmlFile String to the file, where the .pmml of the Randomforest is stored
     */
    public RandomForestWrapper(String pmmlFile) {
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
        this.evaluator = modelEvaluator;
    }


    /**
     * Loads the pixel information from the given picture at picPath into a Map that can be passed to the evaluator
     *
     * @param pic      String of the full path to the file that should be loaded
     * @param loadJson Boolean, to determine if the given file should be interpreted as json or png
     * @return Map of FieldName->FieldValue, containing pixel values from the given png file
     */
    private Map<FieldName, FieldValue> loadInputParameter(String pic, boolean loadJson) {
        float[] picture;
        if (loadJson) {
            picture = TensorflowUtilities.readJsonPic(pic);
        } else {
            picture = TensorflowUtilities.readPic(pic);
        }

        List<InputField> inputFields = this.evaluator.getInputFields(); //{InputField{name=xi, dataType=FLOAT, opType=CONTINUOUS}}; i in int(1,784); if feature_importance of that pixel is 0, the feature is not included. so inputFields.length<784

        Map<FieldName, FieldValue> arguments = new LinkedHashMap<>();
        for (int i = 0; i < inputFields.size(); i++) {
            FieldName inputFieldName = inputFields.get(i).getName();//xi with i in int(1,784)
            int index = Integer.valueOf(inputFieldName.toString().substring(1)); //i in int(1,784)

            Object rawValue = picture[index - 1]; //-1 because indices of DataFields are in int(1,784) and array indices are in int(0,783)
            FieldValue inputFieldValue = inputFields.get(i).prepare(rawValue);
            arguments.put(inputFieldName, inputFieldValue);
        }

        return arguments;
    }

    /**
     * Predicts one bunch of pictures according to Category
     *
     * @param picCat String in ("MNIST"/"Computer"/"Font"/"Handwritten"), describing the category of pictures
     * @param picDir String of directory Data/Own_dat where the pictures are contained
     * @return int[] of predictions for the pictures, where int[0] is the prediction for picture-0
     */
    private int[] getPicCatPredictions(String picCat, String picDir) {
        int[] predictions = new int[10];
        for (int i = 0; i < 10; i++) {
            String path = picDir + picCat + "-" + i + ".json";
            int prediction = predict(path, true);
            predictions[i] = prediction;
        }
        return predictions;
    }

    /**
     * Reads out the result object
     *
     * @param results return value of the evaluator.evaluate() function
     * @return integer representing the predicted number, which is displayed on the given image
     */
    private int readResult(Map<FieldName, ?> results) {
        List<TargetField> targetFields = this.evaluator.getTargetFields();
        TargetField targetField = targetFields.get(0);
        FieldName targetFieldName = targetField.getName();

        //ProbabilityDistribution{result=7.0, probability_entries=[0.0=0.0, 1.0=0.0, 2.0=0.0, 3.0=0.0, 4.0=0.0, 5.0=0.0, 6.0=0.0, 7.0=1.0, 8.0=0.0, 9.0=0.0]}
        ProbabilityDistribution targetFieldValue = (ProbabilityDistribution) results.get(targetFieldName);
        int prediction = ((Double) targetFieldValue.getResult()).intValue();
        return prediction;
    }


    /**
     * Calls the predictions for stored pictures in Data/Own_dat and compares them to the stored
     * Python predictions in the statisticsFile
     *
     * @param statisticsFile String to the .json file where the python predictions are stored
     * @param picDir         directory to the Data/Own_dat folder
     * @return boolean wheter the predictions match
     */
    protected boolean compareResults(String statisticsFile, String picDir) {
        JSONParser parser = new JSONParser();
        Map<String, int[]> picPredictionsJ = new HashMap<>();
        Map<String, int[]> picPredictionsP = new HashMap<>();

        try {
            //read out statistics.json:
            JSONObject obj = (JSONObject) parser.parse(new FileReader(statisticsFile));
            JSONObject picPredictionsJson = (JSONObject) obj.get("picPredictions");

            Object[] picCats = picPredictionsJson.keySet().toArray(); //names of all picture category (MNIST,Font,Computer,Handwritten)
            for (Object picCatObj : picCats) {
                String picCat = (String) picCatObj;//name of the picture category
                JSONArray jsonPredictionsP = (JSONArray) picPredictionsJson.get(picCat);
                int[] predictionsP = new int[10]; //python predictions
                for (int i = 0; i < jsonPredictionsP.size(); i++) {
                    predictionsP[i] = (int) (long) (jsonPredictionsP.get(i));
                }

                //make own prediction on pictures
                int[] predictionsJ = getPicCatPredictions(picCat, picDir);//make java predictions

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

    /**
     * Interface function to be called from outside this class and make a prediction of file at given path
     *
     * @param picPath  String to the path of where the file is located
     * @param loadJson Boolean, wheter to load from a json or png file.
     * @return int which is the prediction, that the loaded RandomForest predicts the picture to display
     */
    protected int predict(String picPath, boolean loadJson) {
        Map<FieldName, FieldValue> inputParameter = loadInputParameter(picPath, loadJson);

        Map<FieldName, ?> results = this.evaluator.evaluate(inputParameter);

        return readResult(results);
    }

}
