import keras
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('../../Data/MNIST_data', one_hot=True)

#unused Function to read Model from seperated model and weight Files
def readModel():
    from keras.models import model_from_json
    with open("./export/model.json") as f:
        json_string = f.read()
    model = model_from_json(json_string)
    model.load_weights('./export/my_model_weights.h5', by_name=True)
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['categorical_accuracy'])
    return model


model = keras.models.load_model("./export/my_model.h5")


loss,accuracity=model.evaluate(mnist.test.images,mnist.test.labels,batch_size=len(mnist.test.images))

print("accuracity on test set: %f %%"%(accuracity*100))
