package RandomForest;

import org.apache.commons.io.FileUtils;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.File;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;
import java.time.LocalDateTime;

public class BenchmarkTest {
    private RandomForestWrapper randomForest;
    private int iterations;
    private long loadingTime;
    private String n_estimators;
    private List<Long> predictionTimes = new LinkedList<>();

    //everything stored in microseconds. HTML print out is in miliseconds
    private double mean;
    private long median;
    private long q75;
    private long q90;
    private long q99;
    private long min;
    private long max;
    private double std;


    public BenchmarkTest(RandomForestWrapper rf, int iterations, String n_estimators, long loadingTime) {
        this.randomForest = rf;
        this.iterations = iterations;
        this.n_estimators = n_estimators;
        this.loadingTime = loadingTime;
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
            long timeDiff = (System.nanoTime() - time0) / 1000; //in microsecond
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
        Arrays.sort(numArray); //ascending
        if (numArray.length % 2 == 0)
            median = (numArray[numArray.length / 2] + numArray[numArray.length / 2 - 1]) / 2;
        else
            median = numArray[numArray.length / 2];

        //q75:
        int index75 = numArray.length - (numArray.length / 4);
        if (numArray.length % 4 == 0)
            q75 = (numArray[index75] + numArray[index75 - 1]) / 2;
        else
            q75 = numArray[index75];
        //q90:
        int index90 = numArray.length - (numArray.length / 10);
        if (numArray.length % 10 == 0)
            q90 = (numArray[index90] + numArray[index90 - 1]) / 2;
        else
            q90 = numArray[index90];
        //q99:
        int index99 = numArray.length - (numArray.length / 100);
        if (numArray.length % 100 == 0)
            q99 = (numArray[index99] + numArray[index99 - 1]) / 2;
        else
            q99 = numArray[index99];


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
                "      <th>90%-quantile</th>" +
                "      <th>99%-quantile</th>" +
                "      <th>min</th>" +
                "      <th>max</th>" +
                "      <th>std deviation</th>" +
                "    </tr>" +
                "  </thead>";
        DecimalFormat f = new DecimalFormat("##0.000");
        System.out.println();
        htmlString += "<tbody>" +
                "    <tr>" +
                "      <td>" + mean/1000 + "</td>" +
                "      <td>" + ((double)median)/1000 + "</td>" +
                "      <td>" + ((double)q90)/1000 + "</td>" +
                "      <td>" + ((double)q99)/1000 + "</td>" +
                "      <td>" + ((double)min)/1000 + "</td>" +
                "      <td>" + ((double)max)/1000 + "</td>" +
                "      <td>" + f.format(std/1000) + "</td>" +
                "    </tr>" +
                "   </tbody></table>";
        htmlString += "<ul>";
        htmlString += "<li>All values in miliseconds</li>";
        htmlString += "<li>Time was measured between having the float[] of an image and getting the prediction output</li>";
        htmlString += "<li>Time needed to load the model from PMML to a jpmml.evaluator object: "+loadingTime+"ms </li>";
        htmlString += "<li>"+iterations+" samples were used to calculate these statistics</li>";
        htmlString += "</ul>";
        htmlString += "</body></html>";

        try {
            FileUtils.writeStringToFile(htmlFile, htmlString);
            System.out.println("Printed benchmark results to: "+filepath);
        } catch (IOException e) {
            e.printStackTrace();
        }

    }
}
