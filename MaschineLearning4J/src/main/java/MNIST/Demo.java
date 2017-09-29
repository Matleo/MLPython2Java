package MNIST;

import net.coobird.thumbnailator.Thumbnails;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.awt.image.Raster;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;

/**
 * Execution Class to load a SavedModel from Tensorflow or Tensorflow.Estimator to recognize the MNSIT data Set, and test it against downloaded pictures
 */
public class Demo {
    private static String importDir1 = "C:/Users/lema/IdeaProjects/Maschine Learning/NeuralNetwork/Estimator/MNISTClassifier/export/";
    private static String importDir2 = "C:/Users/lema/IdeaProjects/Maschine Learning/NeuralNetwork/Tensorflow/CNN/export/";
    private static String importDir3 = "C:/Users/lema/IdeaProjects/Maschine Learning/NeuralNetwork/Tensorflow/Feed Forward NN/SavedModel/export/";

    private static final String picDir = "C:/Users/lema/IdeaProjects/Maschine Learning/NeuralNetwork/Data/Own_dat"; //where the test pics are stored
    private static final String picFile= "/Handwritten"; //which pics to test
    private static final String modelTag = "serve"; //default Tag under which the Metagraph is stored in the SavedModel
    private static boolean estimator = false; //if it is an estimator model


    //which Model to load from file
    public enum Model {
        Estimator_DNN , Tensorflow_CNN , Tensorflow_FFNN
    }

    public static void main(String[] args) throws Exception {
        //decide which Model to load
        Model whichModel = Model.Estimator_DNN;
        String importDir="";
        switch (whichModel){
            case Estimator_DNN: importDir = importDir1;
                break;
            case Tensorflow_CNN: importDir = importDir2;
                break;
            case Tensorflow_FFNN: importDir = importDir3;
                break;
        }

        //if it is an Estimator Model, fetching output and importDir need to be adapted
        if(importDir.contains("/Estimator/")) {
            importDir = IRIS.Demo.getTimestampDir(importDir);
            estimator=true;
        }

        SavedModel model = new SavedModel(importDir,modelTag);

        //run inference
        for(int i=0; i<10;i++){
            float[] inputArray = readPic(picDir+picFile+"-"+i+".png");
            int predict = model.predictNumber(inputArray,estimator);

            System.out.println(i+": Die abgebildete Zahl ist wahrscheinlich eine: "+predict);
        }
    }

    //unused function to load the test pictures from a previously saved json Array
    private static float[] readJsonPic(String path){
        JSONParser parser = new JSONParser();
        double[] arr = new double[784];
        float[] picArr = new float[784];
        try {
            Object obj = parser.parse(new FileReader(path));
            JSONObject jsonObject = (JSONObject) obj;

            JSONArray pic = (JSONArray) jsonObject.get("results");

            for(int i=0;i<pic.size();i++){
                arr[i]=(double)pic.get(i);
            }
            for(int i=0;i<pic.size();i++){
                picArr[i]=(float)arr[i];
            }
        }catch (Exception e){
            e.printStackTrace();
        }
        return picArr;
    }

    /**
     * Function to read a grayscale picture from file
     * @param path the full path of where to find the picture
     * @return float Array with each float in (0,1), where 1 represents black and 0 is white
     */
    private static float[] readPic(String path){
        File imgFile = new File(path);
        float[] imgArr=new float[784];
        try {
            BufferedImage img = ImageIO.read(imgFile);

            img = Thumbnails.of(img).forceSize(28, 28).asBufferedImage();//resize to 28x28
            //Thumbnails.of(img).forceSize(28, 28).toFile("thumbnail.png");//print resized picture

            int width = img.getWidth();
            int height = img.getHeight();
            int[][] imgArrInt = new int[width][height];//alle Pixel eines grayscale IMG in int(0,255)
            Raster raster = img.getData();
            for (int i = 0; i < width; i++) {
                for (int j = 0; j < height; j++) {
                    imgArrInt[i][j] = raster.getSample(j, i, 0);//returns Pixel at (i,j), für getSample indice umgedreht
                }
            }

            //28x28 Array in 784 Vektor pressen
            for(int i=0;i<imgArrInt.length;i++){
                for(int j =0; j<imgArrInt[i].length;j++){
                    int index = i*imgArrInt.length+j;
                    imgArr[index]=(float)imgArrInt[i][j];
                }
            }
            for(int i = 0;i<imgArr.length;i++){
                imgArr[i] = 1-imgArr[i]/255;//von [0,255] auf [0,1] umdrehen
            }

        } catch (IOException e) {
            e.printStackTrace();
        }
        return imgArr;
    }
}