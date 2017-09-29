from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('../../Data/MNIST_data', one_hot=True)

#unused Method to store model and weights seperatly
def saveModel():
    json_string = model.to_json()
    with open("./export/model.json","w") as f:
        f.write(json_string)
    model.save_weights('./export/my_model_weights.h5')

save = True

model = Sequential()
model.add(Dense(550, activation="relu",input_dim=784))
model.add(Dropout(0.75))
model.add(Dense(300, activation="relu"))
model.add(Dropout(0.75))
model.add(Dense(80, activation="relu"))
model.add(Dense(10, activation="softmax"))

model.compile(optimizer="rmsprop",
              loss='categorical_crossentropy',
              metrics=['categorical_accuracy'])

model.fit(mnist.train.images,mnist.train.labels,nb_epoch=10,batch_size=200)

loss,accuracity=model.evaluate(mnist.test.images,mnist.test.labels,batch_size=len(mnist.test.images))

print("accuracity on test set: %f %%"%(accuracity*100))

if save == True:
    model.save("./export/my_model.h5")

