from keras.models import Model
from keras.layers import Input, Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets('../../../../Data/MNIST_data', one_hot=True)




save = True

batch_size = 128
num_classes = 10
epochs = 1

#assume: channel last
input_shape = (28, 28, 1)
#reshaping manually, because dl4J 0.9.1 cannot handle Reshape layer
images_train = mnist.train.images.reshape(mnist.train.images.shape[0], 28, 28, 1)
images_test= mnist.test.images.reshape(mnist.test.images.shape[0], 28, 28, 1)



inputs = Input(shape=(input_shape))
layer1 = Conv2D(nb_filter=32,nb_row=5,nb_col=5, activation='relu')(inputs)
layer2 = MaxPooling2D(pool_size=(2,2))(layer1)
layer3 = Conv2D(nb_filter=64,nb_row=5,nb_col=5, activation='relu')(layer2)
layer4 = MaxPooling2D(pool_size=(2,2))(layer3)#(?,4,4,64)
layer4_flat = Flatten()(layer4)
layer5 = Dense(7 * 7 * 64, activation="relu")(layer4_flat)
layer6 = Dropout(0.5)(layer5)
outputs = Dense(num_classes, activation="softmax")(layer6)




model = Model(input=inputs, output=outputs)

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['categorical_accuracy'])

model.fit(images_train,mnist.train.labels,nb_epoch=epochs,batch_size=batch_size)

loss,accuracity=model.evaluate(images_test,mnist.test.labels,batch_size=len(mnist.test.images))

print("accuracity on test set: %f %%"%(accuracity*100))

if save == True:
    model.save("./export/my_model.h5")
