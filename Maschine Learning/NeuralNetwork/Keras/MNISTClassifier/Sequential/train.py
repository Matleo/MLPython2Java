import json
import tensorflow as tf
import numpy as np
import shutil
import os

from keras.engine import InputLayer
from keras.models import Sequential
from keras.layers import Dense,Dropout,Input

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('../../../../Data/MNIST_data', one_hot=True)

def getPredictions(Pics):
    predictions = []
    for i in range(0,10):
        path = '../../../../Data/Own_dat/'+Pics+'-'+str(i)+'.png'
        file = tf.read_file(path)
        img = tf.image.decode_png(file, channels=1)
        resized_image = tf.image.resize_images(img, [28, 28])
        tensor = tf.reshape(resized_image,[-1])
        tArray=1-sess.run(tensor)/255 #von [0,255] auf [0,1] umdrehen
        input=sess.run(tf.reshape(tArray, [1,784]))

        prediction = model.predict(input)
        pred = int(np.argmax(prediction))
        predictions.append(pred)
    return predictions

def saveConfig():
    export_dir = "./export"
    nd4jFile = "/my_model.h5"
    if os.path.exists(export_dir):
        if os.path.isfile(export_dir+nd4jFile):
            shutil.move(export_dir+nd4jFile,"."+nd4jFile)#move nd4j model and move it into export later
        shutil.rmtree(export_dir)

    signature = tf.saved_model.signature_def_utils.build_signature_def(
        inputs = {'input': tf.saved_model.utils.build_tensor_info(model.input)},
        outputs = {'output': tf.saved_model.utils.build_tensor_info(model.output)},

    )

    builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
    builder.add_meta_graph_and_variables(sess,[tf.saved_model.tag_constants.SERVING],signature_def_map={tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature})
    builder.save()
    if os.path.isfile("."+nd4jFile):
        if os.path.isfile("."+nd4jFile):
            shutil.move("."+nd4jFile,export_dir+nd4jFile)#move nd4j file into export again

    #statistics:
    diction = {}
    diction["epochs"] = epochs
    diction["accuracy"] = round(float(accuracity),4)

    picCategories = ["Handwritten","Computer","MNIST"]
    picDic = {}
    for picCat in picCategories:
        predictions = getPredictions(picCat)
        picDic[picCat] = predictions
    diction["picPredictions"] = picDic
    with open("./export/statistics.json","w") as outfile:
        json.dump(diction,outfile)

    print("\nSaved Configuration to dir: ./%s" % export_dir)



if __name__=="__main__":
    save = True
    epochs = 1
    batch_size = 128

    model = Sequential()
    inputs = Input(shape=(784,), name="input")
    model.add(InputLayer(input_tensor=inputs))
    model.add(Dense(550, activation="relu", ))
    model.add(Dropout(0.75))
    model.add(Dense(300, activation="relu"))
    model.add(Dropout(0.75))
    model.add(Dense(80, activation="relu"))
    model.add(Dense(10, activation="softmax", name="output"))

    model.compile(optimizer="rmsprop",
                  loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])

    model.fit(mnist.train.images,mnist.train.labels,nb_epoch=epochs,batch_size=batch_size)

    loss,accuracity=model.evaluate(mnist.test.images,mnist.test.labels,batch_size=len(mnist.test.images))

    print("accuracity on test set: %f %%"%(accuracity*100))

    if save == True:
        from keras import backend as K
        sess = K.get_session()
        saveConfig()

