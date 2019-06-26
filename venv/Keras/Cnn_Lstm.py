from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Conv1D, MaxPooling1D
from keras.datasets import imdb

# embedding parameter
max_features = 20000
max_len = 100
embedding_size = 128

# cnn parameter
kernel_size = 5
kernel_num = 64
max_pool_size = 4

# lstm parameter

output_size = 70

# training parameter

batch_size = 25

epochs = 100

print('loading data')
(x_train, y_train), (x_val, y_val) = imdb.load_data(num_words=max_features)

print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')


print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print("build a model...")

model = Sequential()
model = model.add(Embedding(max_features, embedding_size, input_length = max_len))

model = model.add(Dropout(0.25))
model = model.add(Conv1D(filters, kernel_size, activation = 'Relu', strides =1))
model = model.add(MaxPooling1D(pool_size = max_pool_size))

model = model.add(LSTM(output_size))
model = model.add(Dense(1))
model = model.add(Activation('sigmoid'))

model = model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

print("Train ......")

model.fit(x_train, y_train, batch_size = batch_size, epochs = epochs, validation_data = (x_val, y_val))
score, accuracy = model.evaluate(x_val, y_val, batch_size = batch_size)

print("Score", score)
print("Accuracy", accuracy)

