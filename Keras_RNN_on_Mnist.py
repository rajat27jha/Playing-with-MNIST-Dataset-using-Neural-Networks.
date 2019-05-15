import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

mnist = tf.keras.datasets.mnist  # taking our model
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# print(x_train.shape)
# print(x_train[0].shape)

x_train = x_train/255.0  # normalizing our data
x_test = x_test/255.0

model = Sequential()
model.add(LSTM(128, input_shape=(x_train.shape[1:]), activation='relu', return_sequences=True))
# return sequences giving data to other layers sequentially or flat
# a dense layer can only accept flattened data but another Recurrent layer wants a sequential data because thats how
# recurrent layer functions
model.add(Dropout(0.2))

model.add(LSTM(128, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(10, activation='softmax'))

opt = tf.keras.optimizers.Adam(lr=1e-3, decay=1e-5)

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

model.fit(x_train, y_train, epocs=3, validation_data=(x_test, y_test))
