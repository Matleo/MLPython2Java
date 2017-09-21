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


public class Demo {
    private static final String importDir = "C:/Users/lema/IdeaProjects/Maschine Learning/NeuralNetwork/CNN/export";
    private static final String picDir = "C:/Users/lema/IdeaProjects/Maschine Learning/NeuralNetwork/Own_dat";
    private static final String modelTag = "s";


    public static void main(String[] args) throws Exception {
        SavedModel model = new SavedModel(importDir,modelTag);
        String picFile= "/Handwritten";

        for(int i=0; i<10;i++){
            float[] inputArray = readJsonPic(picDir+picFile+"-"+i+".json");
            int predict = model.predictNumber(inputArray);

            System.out.println(i+": Die abgebildete Zahl ist wahrscheinlich eine: "+predict);
        }
    }

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


    private static float[] readPic(String path){
        File imgFile = new File(path);
        float[] imgArr=new float[784];
        try {
            BufferedImage img = ImageIO.read(imgFile);

            //img = Thumbnails.of(img).forceSize(28, 28).outputFormat("png").asBufferedImage();//resize to 28x28
            //Thumbnails.of(img).forceSize(28, 28).toFile("thumbnail.png");

            int width = img.getWidth();
            int height = img.getHeight();
            int[][] imgArrInt = new int[width][height];//alle Pixel eines grayscale IMG in int(0,255)
            Raster raster = img.getData();
            for (int i = 0; i < width; i++) {
                for (int j = 0; j < height; j++) {
                    imgArrInt[i][j] = raster.getSample(i, j, 0);//returns Pixel at [i][j]
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