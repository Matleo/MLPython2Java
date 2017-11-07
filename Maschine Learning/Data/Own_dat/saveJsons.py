import json
import tensorflow as tf

if __name__=="__main__":
    picCats = ["Computer", "Font", "Handwritten", "MNIST"]
    for picCat in picCats:
        for i in range(10):
            path = picCat+"-"+str(i)
            file = tf.read_file(path+".png")
            img = tf.image.decode_png(file, channels=1)
            resized_image = tf.image.resize_images(img, [28, 28])
            tensor = tf.reshape(resized_image, [-1])
            with tf.Session() as sess:
                tArray = 1 - sess.run(tensor) / 255  # von [0,255] auf [0,1] umdrehen
                tArray = tArray.tolist()
                for i in range(len(tArray)):
                    tArray[i] = round(tArray[i],8)
                with open(path+".json","w") as file:
                    jsonObj = {}
                    jsonObj["pixelValues"] = tArray
                    json.dump(jsonObj,file)
                    print("Saved: "+path+".json")