package NeuralNetwork.Tensorflow;

import com.google.common.io.Files;
import org.apache.commons.io.IOUtils;
import org.apache.http.HttpEntity;
import org.apache.http.client.methods.CloseableHttpResponse;
import org.apache.http.client.methods.HttpGet;
import org.apache.http.client.methods.HttpPost;
import org.apache.http.entity.ContentType;
import org.apache.http.entity.StringEntity;
import org.apache.http.impl.client.CloseableHttpClient;
import org.apache.http.impl.client.HttpClients;
import org.apache.http.util.EntityUtils;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.awt.image.Raster;
import java.awt.image.WritableRaster;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;

public class InferenceClient {
    private static final String picDir = "../Maschine Learning/Data/Own_dat/";
    private static final String Local_URI = "http://localhost:8000/predict";
    private static final String Heroku_URI = "https://stark-savannah-40830.herokuapp.com/predict";

    public static void main(String[] args) {
        String Server_URI = Heroku_URI;
        String picFile = "Font-2.png";//default
        if(args.length>0){
            if(!args[0].contains(".png"))args[0]=args[0]+".png";
            File f = new File(picDir+args[0]);
            if(f.exists() && !f.isDirectory() && Files.getFileExtension(picDir+args[0]).equals("png")) {
                picFile=args[0];
                System.out.println("Infering the webservice with given picture at: "+args[0]);
            }else{
                System.out.println("The argument that you passed is invalid.");
                System.out.println("It must be a .png file, which is contained in directory: "+picDir);
                System.out.println("\nInfering the webservice with default picture: "+picFile);
            }
        }else{
            System.out.println("Infering the webservice with default picture: "+picFile);
        }
        String path = picDir + picFile;
        int[][] picture = readPic(path);

        CloseableHttpClient httpclient = HttpClients.createDefault();

        //Define request: pass int array representing the picture
        HttpPost httpPost = new HttpPost(Server_URI);
        JSONObject json = new JSONObject();
        JSONArray picArray = new JSONArray();
        for (int i = 0; i < picture.length; i++) {
            JSONArray subArr = new JSONArray();
            for (int j = 0; j < picture[i].length; j++) {
                subArr.add(picture[i][j]);
            }
            picArray.add(subArr);
        }
        json.put("picArray", picArray);
        String JSON_STRING = json.toJSONString();
        StringEntity requestEntity = new StringEntity(
                JSON_STRING,
                ContentType.APPLICATION_JSON);
        httpPost.setEntity(requestEntity);

        CloseableHttpResponse response = null;
        try {
            //get response from server
            response = httpclient.execute(httpPost);
        } catch (IOException e) {
            e.printStackTrace();
        }

        try {
            //read out the response
            System.out.println("Response status line: "+response.getStatusLine());
            HttpEntity entity = response.getEntity();
            InputStream is = entity.getContent();
            String content = IOUtils.toString(is, StandardCharsets.UTF_8);
            System.out.println("Response content: "+content);
            EntityUtils.consume(entity);
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            try {
                response.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    /**
     * Unused function. Prints given pixels to "saved.png" file at project root
     * @param pixels2d int[height][width] array, containing grayscale values
     */
    private static void printPicArray(int[][] pixels2d) {
        int[] pixelsFlat = Arrays.stream(pixels2d)
                .flatMapToInt(Arrays::stream)
                .toArray();

        BufferedImage image = new BufferedImage(pixels2d[0].length, pixels2d.length, BufferedImage.TYPE_BYTE_GRAY);
        WritableRaster wr = image.getRaster() ;
        for (int y=0, count=0 ; y < image.getHeight() ; y++)
            for (int x=0 ; x < image.getWidth() ; x++, count++)
                wr.setSample(x, y, 0, pixelsFlat[count]) ;

        try {
            File outputfile = new File("./saved.png");
            ImageIO.write(image, "png", outputfile);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * Reads the grayscale integers of image at given path
     * @param path path to image to read pixels
     * @return integer values in array of shape: [height][width]
     */
    private static int[][] readPic(String path) {
        File imgFile = new File(path);
        int[][] imgArrInt = new int[0][];
        try {
            BufferedImage img = ImageIO.read(imgFile);
            int width = img.getWidth();
            int height = img.getHeight();
            imgArrInt = new int[height][width];//alle Pixel eines grayscale IMG in int(0,255)
            Raster raster = img.getData();
            for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                    imgArrInt[i][j] = raster.getSample(j, i, 0);//because original pic is width x heigth
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return imgArrInt;
    }

    private static void exampleGetRequest(CloseableHttpClient httpclient) {
        HttpGet httpGet = new HttpGet("http://localhost:8000");
        CloseableHttpResponse response = null;
        try {
            response = httpclient.execute(httpGet);
        } catch (IOException e) {
            e.printStackTrace();
        }

        try {
            System.out.println(response.getStatusLine());
            HttpEntity entity = response.getEntity();
            InputStream is = entity.getContent();
            String content = IOUtils.toString(is, StandardCharsets.UTF_8);
            System.out.println(content);
            is.close();

            EntityUtils.consume(entity);//frees the ressource
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            try {
                response.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }
}

