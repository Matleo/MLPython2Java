import keras
import numpy as np
from flask import Flask
import tensorflow as tf

#assume 0 = negative and 1=positive
def predictRating(example):
    #prediction is in (0,1) giving a prediction depending if it is closer to 1 or 0
    prediction = model.predict(example)
    if prediction[0][0] - 0.5 > 0:
        return "The given rating is probably--> positive"
    else:
        return "The given rating is probably--> negative"

def predictions():
    model = keras.models.load_model("./export/sequential.h5")
    example1 = np.array([[15,256, 4, 2, 7, 3766, 5,723, 36, 71, 43,530
                             ,476, 26,400,317, 46, 7, 4, 12118, 1029, 13,104, 88
                             ,4,381, 15,297, 98, 32, 2071, 56, 26,141, 6,194
                             ,7486, 18, 4,226, 22, 21,134,476, 26,480, 5,144
                             ,30, 5535, 18, 51, 36, 28,224, 92, 25,104, 4,226
                             ,65, 16, 38, 1334, 88, 12, 16,283, 5, 16, 4472,113
                             ,103, 32, 15, 16, 5345, 19,178, 32]])
    example0 = np.array([[125,68 ,2 ,6853 ,15 ,349 ,165 ,4362 ,98 ,5 ,4 ,228,9,43,2,
                          1157,15,299,120,5,120,174,11,220,175,136,50,9,4373,228,
                          8255,5,2,656,245,2350,5,4,9837,131,152,491,18,2,32,
                          7464,1212,14,9,6,371,78,22,625,64,1382,9,8,168,145,
                          23,4,1690,15,16,4,1355,5,28,6,52,154,462,33,89,
                          78,285,16,145,95]])
    predictRating(example1)
    return(predictRating(example0))


app=Flask(__name__)
model = keras.models.load_model("./export/sequential.h5")
graph = tf.get_default_graph()

@app.route("/")
def hello_world():
    return "Hello World!"

@app.route("/predict")
def predict():
    return predictions()




if __name__ == "__main__":
    app.run(debug=True,port=8000)
