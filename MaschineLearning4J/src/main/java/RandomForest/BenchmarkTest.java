package RandomForest;

import org.apache.commons.io.FileUtils;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;
import java.time.LocalDateTime;

public class BenchmarkTest {
    private RandomForestWrapper randomForest;
    private int iterations;
    private String n_estimators;
    private List<Long> predictionTimes = new LinkedList<>();

    //everything in microseconds
    private double mean;
    private long median;
    private long min;
    private long max;
    private double std;


    public BenchmarkTest(RandomForestWrapper rf, int iterations, String n_estimators) {
        this.randomForest = rf;
        this.iterations = iterations;
        this.n_estimators = n_estimators;
    }

    public void run() {
        System.out.println("\nRunning benchmark test now...");
        DataSetIterator mnistTest = null;
        try {
            mnistTest = new MnistDataSetIterator(1, false, 123);
        } catch (IOException e) {
            e.printStackTrace();
        }

        for (int i = 0; i < iterations; i++) {
            DataSet next = mnistTest.next(); //(1,784)
            float[] sample = getFloatArray(next);

            long time0 = System.nanoTime();
            int prediction = randomForest.predict(sample);
            long timeDiff = (System.nanoTime() - time0) / 1000; //in microsekunde
            predictionTimes.add(timeDiff);
        }

        calculateStatistics();
        printStatisticsToHtml();
    }

    private float[] getFloatArray(DataSet ds) {
        INDArray next2 = ds.getFeatureMatrix();//(1,784)
        float[] array = new float[next2.shape()[1]];
        //convert INDArray to float[]:
        for (int j = 0; j < next2.shape()[1]; j++) {
            array[j] = next2.getFloat(j);
        }
        return array;
    }

    private void calculateStatistics() {
        //mean:
        long sum = 0;
        for (Long time : predictionTimes) {
            sum += time;
        }
        mean = sum / predictionTimes.size();

        //median:
        Long[] numArray = new Long[predictionTimes.size()];
        predictionTimes.toArray(numArray);
        Arrays.sort(numArray);
        if (numArray.length % 2 == 0)
            median = (numArray[numArray.length / 2] + numArray[numArray.length / 2 - 1]) / 2;
        else
            median = numArray[numArray.length / 2];

        //min max:
        min = Collections.min(predictionTimes);
        max = Collections.max(predictionTimes);

        //std deviation:
        double sumSquares = 0.0;
        for (Long time : predictionTimes) {
            long diff = time - (long) mean;
            double square = Math.pow((double) diff, 2);
            sumSquares += square;
        }
        double variance = sumSquares / (predictionTimes.size() - 1);
        std = Math.sqrt(variance);


    }

    private void printStatisticsToHtml() {
        String filepath = "src/main/java/RandomForest/benchmark_" + n_estimators + ".html";
        File htmlFile = new File(filepath);
        String htmlString = "<html><body>";
        htmlString += "<h1>Benchmark test " + LocalDateTime.now() + "</h1>";
        htmlString += "<h2>SequentialLoadBenchmark</h2>";
        htmlString += "<table border=\"1\" class=\"dataframe\">";
        htmlString += "<thead>" +
                "    <tr style=\"text-align: right;\">" +
                "      <th>mean</th>" +
                "      <th>median</th>" +
                "      <th>min</th>" +
                "      <th>max</th>" +
                "      <th>std deviation</th>" +
                "    </tr>" +
                "  </thead>";
        htmlString += "<tbody>" +
                "    <tr>" +
                "      <td>" + (int) mean + "</td>" +
                "      <td>" + median + "</td>" +
                "      <td>" + min + "</td>" +
                "      <td>" + max + "</td>" +
                "      <td>" + (int) std + "</td>" +
                "    </tr>" +
                "   </tbody></table>";
        htmlString += "<p>(All values in microseconds)</p>";
        htmlString += "</body></html>";

        try {
            FileUtils.writeStringToFile(htmlFile, htmlString);
            System.out.println("Printed benchmark results to: MaschineLearning4J/"+filepath);
        } catch (IOException e) {
            e.printStackTrace();
        }

    }
}
