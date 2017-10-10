from flask import Flask
import tensorflow as tf

app=Flask(__name__)



@app.route("/")
def hello_world():
    return "Hello World!"

@app.route("/predict")
def predict():
    picName="MNIST"

    sess = tf.Session()
    import_dir = "./export"
    tf.saved_model.loader.load(sess, ["serve"], import_dir)

    graph = tf.get_default_graph()
    y3 = graph.get_tensor_by_name("output:0")
    x = graph.get_tensor_by_name("input:0")
    dKeep = graph.get_tensor_by_name("dropoutRate:0")
    return predictPic(picName,0,sess,y3,x,dKeep)



def predictPic(picName, i,sess,y3,x,dKeep):
    path = '../../../../../Data/Own_dat/'+picName+'-'+str(i)+'.png'
    file = tf.read_file(path)
    img = tf.image.decode_png(file, channels=1)
    resized_image = tf.image.resize_images(img, [28, 28])
    tensor=tf.reshape(resized_image, [-1])
    tArray=1-sess.run(tensor)/255 #von [0,255] auf [0,1] umdrehen
    output=sess.run(tf.reshape(tArray, [1,784]))
    guessed= sess.run(y3, feed_dict={x:output,dKeep:1})
    guessedIndex= sess.run(tf.argmax(y3,1), feed_dict={x:output,dKeep:1})
    guessedIndex=list(guessedIndex)[0]#um von set auf int zu kommen
    guessedProb= guessed[0][guessedIndex]*100

    return "%i: Die abgebildete Zahl ist zu %f%% eine: %d." % (i,guessedProb,guessedIndex)



if __name__ == "__main__":
    app.run(debug=True,port=8000)
