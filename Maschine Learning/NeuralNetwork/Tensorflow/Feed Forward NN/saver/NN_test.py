import tensorflow as tf

def printPredictions(Pics):
    for i in range(0,10):
        file = tf.read_file('../../../Data/Own_dat/'+Pics+'-'+str(i)+'.png')
        img = tf.image.decode_png(file, channels=1)
        resized_image = tf.image.resize_images(img, [28, 28])
        tensor=tf.reshape(resized_image, [-1])
        tArray=1-sess.run(tensor)/255 #von [0,255] auf [0,1] umdrehen
        determinNumber(tArray,i)

def determinNumber(tArray,i):
    output=sess.run(tf.reshape(tArray, [1,784]))
    guessed= sess.run(y3, feed_dict={x:output,dKeep:1})
    guessedIndex= sess.run(tf.argmax(y3,1), feed_dict={x:output,dKeep:1})
    guessedIndex=list(guessedIndex)[0]#um von set auf int zu kommen
    guessedProb= guessed[0][guessedIndex]*100
    print("%i: Die abgebildete Zahl ist zu %f%% eine: %d." % (i,guessedProb,guessedIndex))



import_dir = "./export/model"

sess = tf.Session()
saver = tf.train.import_meta_graph(import_dir+".meta")
saver.restore(sess, import_dir)

graph = tf.get_default_graph()
y3 = graph.get_tensor_by_name("output:0")
x = graph.get_tensor_by_name("input:0")
dKeep = graph.get_tensor_by_name("dropoutRate:0")

printPredictions("MNIST")