package RandomForest;

import org.dmg.pmml.FieldName;
import org.dmg.pmml.PMML;
import org.jpmml.evaluator.*;
import org.xml.sax.SAXException;

import javax.xml.bind.JAXBException;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.InputStream;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public class IRISClassifier {
    public final static String file = "../Maschine Learning/RandomForest/MaaS/RandomForestIris.pmml";
    public final static float[] IRISSample = {4.5f, 1.7f, 4.9f, 2.5f};//Iris-virginica


    public static void main(String[] args) {

        Evaluator evaluator = loadEvaluator();

        Map<FieldName, FieldValue> inputParameter = loadInputParameter(evaluator);

        Map<FieldName, ?> results = evaluator.evaluate(inputParameter);

        printResults(results, evaluator);
    }

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

    private static Map<FieldName, FieldValue> loadInputParameter(Evaluator evaluator) {
        Map<FieldName, FieldValue> arguments = new LinkedHashMap<>();
        List<InputField> inputFields = evaluator.getInputFields();
        for (int i = 0; i < inputFields.size(); i++) {
            FieldName inputFieldName = inputFields.get(i).getName();
            Object rawValue = IRISSample[i];
            FieldValue inputFieldValue = inputFields.get(i).prepare(rawValue);
            arguments.put(inputFieldName, inputFieldValue);
        }
        return arguments;
    }

    private static void printResults(Map<FieldName, ?> results, Evaluator evaluator){
        List<TargetField> targetFields = evaluator.getTargetFields();
        for (TargetField targetField : targetFields) {
            FieldName targetFieldName = targetField.getName();

            ProbabilityDistribution targetFieldValue = (ProbabilityDistribution)results.get(targetFieldName);
            System.out.println(targetFieldValue.getResult());
        }
    }
}
