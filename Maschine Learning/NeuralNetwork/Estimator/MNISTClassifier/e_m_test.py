import os
import numpy as np
import tensorflow as tf
def printPredictions(pics):
    for i in range(0,10):
        path = '../../Data/Own_dat/'+pics+'-'+str(i)+'.png'
        file = tf.read_file(path)
        img = tf.image.decode_png(file, channels=1)
        resized_image = tf.image.resize_images(img, [28, 28])
        tensor=tf.reshape(resized_image, [-1])
        with tf.Session() as sess:
            tArray=1-sess.run(tensor)/255 #von [0,255] auf [0,1] umdrehen
        predictNumber(tArray,i)

def predictNumber(tArray, i):
    predictions = predict_fn({"input":[tArray]})
    prediction = np.argmax(predictions["scores"][0])
    print("%i: Die abgebildete Zahl ist zu wahrscheinlich eine: %d." % (i,prediction))


import_super_dir = "./export/"
timestamp=os.listdir(import_super_dir)[0] #first entry in directory
import_dir = import_super_dir+timestamp

predict_fn = tf.contrib.predictor.from_saved_model(import_dir)



printPredictions("MNIST")