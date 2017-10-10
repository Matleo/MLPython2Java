import json
import os
import shutil
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("../../../../Data/MNIST_data/", one_hot=True)

def getPredictions(Pics):
    predictions = []
    for i in range(0,10):
        path = '../../../../../Data/Own_dat/'+Pics+'-'+str(i)+'.png'
        file = tf.read_file(path)
        img = tf.image.decode_png(file, channels=1)
        resized_image = tf.image.resize_images(img, [28, 28])
        tensor=tf.reshape(resized_image, [-1])
        tArray=1-sess.run(tensor)/255 #von [0,255] auf [0,1] umdrehen

        pred = determinNumber(tArray)
        predictions.append(pred)
    return predictions

def determinNumber(tArray):
    output=sess.run(tf.reshape(tArray, [1,784]))
    guessed= sess.run(y3, feed_dict={x:output,dKeep:1})
    guessedIndex= sess.run(tf.argmax(y3,1), feed_dict={x:output,dKeep:1})
    guessedIndex=list(guessedIndex)[0]#um von set auf int zu kommen
    return int(guessedIndex)

def saveConfig():
    export_dir = "./export"
    if os.path.exists(export_dir):
        shutil.rmtree(export_dir)

    signature = tf.saved_model.signature_def_utils.build_signature_def(
        inputs = {'input': tf.saved_model.utils.build_tensor_info(x)},
        outputs = {'output': tf.saved_model.utils.build_tensor_info(y3)},
    )

    builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
    builder.add_meta_graph_and_variables(sess,[tf.saved_model.tag_constants.SERVING],signature_def_map={tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature})
    builder.save()

    #statistics:
    diction = {}
    diction["steps"] = int(steps)
    diction["accuracy"] = round(float(acc),4)

    picCategories = ["Handwritten","Computer","MNIST"]
    picDic = {}
    for picCat in picCategories:
        predictions = getPredictions(picCat)
        picDic[picCat] = predictions
    diction["picPredictions"] = picDic;
    with open("./export/statistics.json","w") as outfile:
        json.dump(diction,outfile)

    print("\nSaved Configuration to dir: ./%s" % export_dir)


save = False
#----------------------------------------------------------------------------------------------------------


#images: beliebig viele(None) Bilder, wobei jedes auf 784 Pixel abgebildet wurde. Jeder Pixel ist in (0,1), wobei 1 dunkel ist
x = tf.placeholder(tf.float32, [None, 784],name="input")

#weights: aus der MM x*W soll pro Bild ein one-hot vector(dim 10) entstehen
W1 = tf.Variable(tf.truncated_normal([784, 250],stddev=0.1))
b1 = tf.Variable(tf.zeros([250]))  #bias, wird später zu den one-hot vectors addiert

W2 = tf.Variable(tf.truncated_normal([250, 70],stddev=0.1))
b2 = tf.Variable(tf.zeros([70]))

W3 = tf.Variable(tf.truncated_normal([70, 10],stddev=0.1))
b3 = tf.Variable(tf.zeros([10]))

dKeep=tf.placeholder(tf.float32, name="dropoutRate")
#assumed labels: softmax normalisiert(so dass Summe=1) die Aussagestärke, die sich durch die gewichtetete Summe aus x*W ergibt
y1d = tf.nn.relu(tf.matmul(x, W1) + b1) #hat Dimension [NONE,10]. enthält Wahrscheinlichkeiten dafür, welche Zahl repräsentiert wurde
y1=tf.nn.dropout(y1d,dKeep)
y2d = tf.nn.relu(tf.matmul(y1, W2) + b2) #hat Dimension [NONE,10]. enthält Wahrscheinlichkeiten dafür, welche Zahl repräsentiert wurde
y2=tf.nn.dropout(y2d,dKeep)

y3 = tf.nn.softmax(tf.matmul(y2,W3) + b3)
y3=tf.identity(y3,"output")
y_ = tf.placeholder(tf.float32, [None, 10]) #actual labels
y_=tf.identity(y_,"y_")


#measures how inefficient the predictions are
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y3, labels=y_)
cross_entropy = tf.reduce_mean(cross_entropy)*100


#asking tf to minimize cross_entropy using gradient descent
eta = 0.01
optimizer = tf.train.GradientDescentOptimizer(eta)
train_step = optimizer.minimize(cross_entropy)
#adds new operations to the computation graph.
#These operations included ones to compute gradients, compute parameter update steps, and apply update steps to the parameters.
#The returned operation train_step, when run, will apply the gradient descent updates to the parameters.
#Training the model can therefore be accomplished by repeatedly running train_step.


sess = tf.InteractiveSession() #launch the model
tf.global_variables_initializer().run() #initialize the variables


#Vector aus boolean, welcher speichert ob prediction für Bild i korrekt war
correct_prediction = tf.equal(tf.argmax(y3,1), tf.argmax(y_,1))#argmax gives index of highest entry(also jeweils wo die 1 aus one-hot vector)
#berechnet aus boolean Vektor, die Treffersicherheit
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


#start training
steps =10000
for i in range(steps):
    batch_xs, batch_ys = mnist.train.next_batch(100) #gib mir 100 x,y aus den Daten(für Effizienz)
    #feed_dict replaces the placeholder tensors x and y_ with the training examples
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, dKeep:0.75})

    #aktuelle Genauigkeit und loss
    if i % 500 == 0:
        print("step "+str(i)+":", end='')
        print(sess.run([accuracy,cross_entropy], feed_dict={x: mnist.train.images, y_: mnist.train.labels, dKeep:1.0}))

acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels, dKeep:1.0})
print('test accuracy %g' % acc)



#----------------------------------------------------------------------------------------------------------
if save == True:
    saveConfig()

