from keras.preprocessing import sequence
from keras.models import Model
from keras.layers import Dense, Embedding, Input
from keras.layers import LSTM
from keras.datasets import imdb

if __name__=="__main__":
    save = True
    max_features = 20000 #the 20 000 most popular words will be regarded (total of 88585 Words)
    maxlen = 80  # cut texts after this number of words (among top max_features most common words)
    batch_size = 50
    epochs = 1


    print('Loading data...')
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
    print(len(x_train), 'train sequences')
    print(len(x_test), 'test sequences')
    print('Pad sequences (samples x time)')
    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)

    print('Build model...')
    inputs = Input(shape=(maxlen,))
    layer1 = Embedding(max_features, 128)(inputs)
    layer2 = LSTM(128)(layer1)
    outputs = Dense(1, activation='sigmoid')(layer2)

    model = Model(inputs=inputs, outputs=outputs)


    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    #print(model.summary())

    print('Train...')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test))
    score, acc = model.evaluate(x_test, y_test,
                                batch_size=batch_size)
    print('Test score:', score)
    print('Test accuracy:', acc)

    if save == True:
        export_dir = "./export"
        import os
        import shutil
        if os.path.exists(export_dir):
            shutil.rmtree(export_dir)

        import keras.backend as K
        import tensorflow as tf
        sess = K.get_session()

        signature = tf.saved_model.signature_def_utils.build_signature_def(
            inputs={'input': tf.saved_model.utils.build_tensor_info(model.input)},
            outputs={'output': tf.saved_model.utils.build_tensor_info(model.output)}
        )
        signatureDef = {tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature}

        builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
        builder.add_meta_graph_and_variables(sess,[tf.saved_model.tag_constants.SERVING],signature_def_map=signatureDef)
        builder.save()

        model.save("./export/model.h5")
