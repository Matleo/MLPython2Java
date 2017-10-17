from flask import Flask, request
import tensorflow as tf

app = Flask(__name__)

@app.route("/predict", methods=['POST'])
def predict():
    req = request.get_json()
    picArray = req["picArray"]
    reshaped_Array = reshapePic(picArray)

    return predictPic(reshaped_Array)


@app.route("/example", methods=['GET', 'POST'])
def example():
    req = request.get_json()
    list = req["example"]
    ex = ''.join(str(list))
    return ex


@app.route("/")
def predict_example():
    picName = "MNIST"
    i = 0
    path = '../../../../../Data/Own_dat/' + picName + '-' + str(i) + '.png'
    file = tf.read_file(path)
    pic = tf.image.decode_png(file, channels=1)
    reshaped_Array = reshapePic(pic)
    return predictPic(reshaped_Array)


def predictPic(picArray):
    sess = tf.Session()
    import_dir = "./export"
    tf.saved_model.loader.load(sess, ["serve"], import_dir)
    graph = tf.get_default_graph()
    y = graph.get_tensor_by_name("output:0")
    x = graph.get_tensor_by_name("input:0")
    dKeep = graph.get_tensor_by_name("dropoutRate:0")

    guessed = sess.run(y, feed_dict={x: picArray, dKeep: 1})
    guessedIndex = sess.run(tf.argmax(y, 1), feed_dict={x: picArray, dKeep: 1})
    guessedIndex = list(guessedIndex)[0]  # um von set auf int zu kommen
    guessedProb = guessed[0][guessedIndex] * 100

    return "Die abgebildete Zahl ist zu %f%% eine: %d." % (guessedProb, guessedIndex)

def reshapePic(pic):
    for i in range(len(pic)):
        for j in range(len(pic[i])):
            pic[i][j] = [pic[i][j]]
    resized_image = tf.image.resize_images(pic, [28, 28])
    tensor = tf.reshape(resized_image, [-1])

    sess = tf.Session()
    tArray = 1 - sess.run(tensor) / 255  # von [0,255] auf [0,1] umdrehen
    reshaped_Array = sess.run(tf.reshape(tArray, [1, 784]))
    return reshaped_Array

if __name__ == "__main__":
    app.run(debug=True, port=8000)
