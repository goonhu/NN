# cnn for text classification sample

# embedding_dim = 50
# filter_size = (3, 8)
# num_filter = 10
# dropout = (0.5, 0.8) respectively
# hidden_dim = 50


import numpy as np

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Convolution1D, MaxPooling1D, Input, Embedding
from keras.layers.merge import concatenate


from keras.preprocessing import sequence


embedding_dim = 50
filter_sizes = (3, 8)
# defined as tuples so can't be adjusted
num_filters = 10
dropout_ = (0,5, 0.8)
# defined as tuples so can't be adjusted
hidden_dim = 50

batch_size = 64
num_epochs = 10

# ------------------------------this is a step for data preprocessing -------------------------------
sequence_length = 400
# fix for the length of input to 400
max_words = 500
# fix for the num of words in a sentence to 500


# build model

input_tensor = Input(shape= input_shape)
# initialize the input tensor for the model, USING keras.layer.Input

z = Embedding(input_dim = len(vocabulary), output_dim = embedding_dim, input_length = sequence_length, name = "embedding")
z = Dropout(dropout_[0])(z)

# convolutional block
conv_blocks = []

for k_s in filter_sizes:
    conv = Convolution1D(filters = num_filters,
                         kernel_size = k_s,
                         strides = 1,
                         padding = 'valid',
                         activation = 'relu')(z)
    conv = MaxPooling1D(pool_size = 2, strides = None, padding = 'valid')(conv)
    conv = Flatten()(conv)
    conv_blocks.append(conv)

z = concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]

z = Dropout(dropout_[1])(z)
z = Dense(hidden_dim, activation = 'relu')(z)
model_output = Dense(hidden_dim = 1, activation = 'sigmoid')(z)
# last layer for classification : sigmoid for binary classification

model = Model(input, model_output)
model.compile(loss = 'binary_crossentrophy', optimizer = 'adam', metrics = ["accuracy"])

# model weight initialisation with embedding weight

weights = np.array([v for v in embedding_weights])
# embedding_weights : this means hidden layer weights, in other words look up table ?
embedding_layer = moddel.get_layer("embedding")
embedding_layer.set_weights([weights])


model.fit(x_train, y_train, batch_size = batch_size,
          epochs = num_epochs, validation_data = (x_test, y_test),
          verbose = 0)




