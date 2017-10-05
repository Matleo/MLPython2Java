from keras.preprocessing import sequence
from keras.models import Model
from keras.layers import Dense, Embedding, Input
from keras.layers import LSTM
from keras.datasets import imdb

save = True
max_features = 20000 #the 20 000 most popular words will be regarded (total of 88585 Words)
maxlen = 80  # cut texts after this number of words (among top max_features most common words)
batch_size = 50
epochs = 1
input_shape = 80


print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(nb_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')
print(x_train[0])
print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Build model...')
inputs = Input(batch_shape=(batch_size,input_shape))
layer1 = Embedding(max_features, 128)(inputs)
layer2 = LSTM(128)(layer1)
outputs = Dense(1, activation='sigmoid')(layer2)

model = Model(input=inputs, output=outputs)


model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
print(model.summary())

print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          nb_epoch=epochs,
          validation_data=(x_test, y_test))
score, acc = model.evaluate(x_test, y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)

if save == True:
    model.save("./export/model.h5")