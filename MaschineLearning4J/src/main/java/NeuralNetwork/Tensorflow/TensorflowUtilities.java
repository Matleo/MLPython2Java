package NeuralNetwork.Tensorflow;

import net.coobird.thumbnailator.Thumbnails;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.tensorflow.Tensor;
import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.awt.image.Raster;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.nio.FloatBuffer;
import java.util.Arrays;
import java.util.Map;

/**
 * Utility functions to work with Tensorflow API
 */
public class TensorflowUtilities {

    /**
     * Converts a float Array to a Tensor
     * @param array array to convert to a Tensor
     * @return Tensor representing the Array
     */
    protected static Tensor toTensor(float[] array) {
        long[] shape = {1, 784};
        FloatBuffer fb = FloatBuffer.wrap(array);
        Tensor t = Tensor.create(shape, fb);
        return t;
    }

    /**
     * Converts a Tensor to a float Array
     * @param t Tensor to convert
     * @return Array representing the Tensor
     */
    protected static float[] toArray(Tensor t) {
        FloatBuffer fb = FloatBuffer.allocate(t.numElements());
        t.writeTo(fb);
        return fb.array();
    }


    /**
     * @param array array to iterate over to find max value
     * @return index of maximum value in array
     */
    protected static int maxIndex(float[] array) {
        float max = Float.MIN_VALUE;
        int maxIndex = -1;
        for (int i = 0; i < array.length; i++) {
            if (array[i] > max) {
                max = array[i];
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    /**
     * Function to load the test pictures from a previously saved json Array
     * @param path the full path of where to find the picture
     * @return float Array with each float in (0,1), where 1 represents black and 0 is white
     */
    protected static float[] readJsonPic(String path) {
        JSONParser parser = new JSONParser();
        float[] picArr = new float[784];
        try {
            Object obj = parser.parse(new FileReader(path));
            JSONObject jsonObject = (JSONObject) obj;

            JSONArray pic = (JSONArray) jsonObject.get("results");

            for (int i = 0; i < pic.size(); i++) {
                picArr[i] = ((Double) pic.get(i)).floatValue();
            }

        } catch (Exception e) {
            e.printStackTrace();
        }
        return picArr;
    }

    /**
     * function to read a grayscale picture from file
     * @param path the full path of where to find the picture
     * @return float Array with each float in (0,1), where 1 represents black and 0 is white
     */
    protected static float[] readPic(String path) {
        File imgFile = new File(path);
        float[] imgArr = new float[784];
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
            for (int i = 0; i < imgArrInt.length; i++) {
                for (int j = 0; j < imgArrInt[i].length; j++) {
                    int index = i * imgArrInt.length + j;
                    imgArr[index] = (float) imgArrInt[i][j];
                }
            }
            for (int i = 0; i < imgArr.length; i++) {
                imgArr[i] = 1 - imgArr[i] / 255;//von [0,255] auf [0,1] umdrehen
            }

        } catch (IOException e) {
            e.printStackTrace();
        }
        return imgArr;
    }


    /**
     * Compares two maps of String->int[] to evaluate if they are equal
     * @param mapJ map containing Java predictions
     * @param mapP map containing Python predictions
     * @return returns whether the maps contain equal content
     */
    protected static boolean compareMaps(Map<String, int[]> mapJ, Map<String, int[]> mapP) {
        System.out.println("\n\nComparing Java and Python picture predictions...");
        boolean match = true;
        if(mapJ.size()!=mapP.size()){
            System.out.println("The size of the maps don't match!");
            return false;
        }

        for (String key : mapJ.keySet()) {
            int[] predJ = mapJ.get(key);
            int[] predP = mapP.get(key);
            for (int i = 0; i < mapJ.get(key).length; i++) {
                if (predJ[i] != predP[i]) match=false;
            }
        }

        if (match) {
            System.out.println("***Success***");
            System.out.println("The python and java predictions match!");
            return true;
        } else {
            System.out.println("***Failure***");
            System.out.println("The python and java predictions dont match");
            System.out.println("Printing out the prediction...");
            for (String key : mapJ.keySet()) {
                System.out.println("Category \"" + key + "\":");
                System.out.println("    Java   :" + Arrays.toString(mapJ.get(key)));
                System.out.println("    Python :" + Arrays.toString(mapP.get(key)));
            }
            return false;
        }
    }



}
