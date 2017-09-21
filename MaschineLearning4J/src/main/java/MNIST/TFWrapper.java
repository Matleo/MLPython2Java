package MNIST;


import org.tensorflow.Tensor;


import java.nio.FloatBuffer;

public class TFWrapper {


    public static Tensor toTensor(float[] array){
        long[] shape = {1,784};
        FloatBuffer fb = FloatBuffer.wrap(array);
        Tensor t = Tensor.create(shape,fb);

        return t;
    }
    public static float[] toArray(Tensor t){
        FloatBuffer fb = FloatBuffer.allocate(t.numElements());
        t.writeTo(fb);
        return fb.array();
    }

    public static void printTensor(Tensor t){
        for (float f:toArray(t)) {
            System.out.println(f);
        }
    }

    public static int maxIndex(float[] array){
        float max = Float.MIN_VALUE;
        int maxIndex = -1;
        for (int i=0;i<array.length;i++) {
            if(array[i]>max) {
                max=array[i];
                maxIndex=i;
            }
        }
        return maxIndex;
    }


}
