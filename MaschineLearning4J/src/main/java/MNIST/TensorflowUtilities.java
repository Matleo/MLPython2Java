package MNIST;

import org.tensorflow.Tensor;

import java.nio.FloatBuffer;

/**
 * Utility functions to work with Tensorflow API
 */
public class TensorflowUtilities {

    /**
     * Converts a float Array to a Tensor
     * @param array array to convert to a Tensor
     * @return Tensor representing the Array
     */
    public static Tensor toTensor(float[] array){
        long[] shape = {1,784};
        FloatBuffer fb = FloatBuffer.wrap(array);
        Tensor t = Tensor.create(shape,fb);

        return t;
    }

    /**
     * Converts a Tensor to a float Array
     * @param t Tensor to convert
     * @return Array representing the Tensor
     */
    public static float[] toArray(Tensor t){
        FloatBuffer fb = FloatBuffer.allocate(t.numElements());
        t.writeTo(fb);
        return fb.array();
    }

    /**
     * Function to print a Tensors values
     * @param t Tensor to print
     */
    public static void printTensor(Tensor t){
        for (float f:toArray(t)) {
            System.out.println(f);
        }
    }

    /**
     * @param array array to iterate over to find max value
     * @return index of maximum value in array
     */
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
