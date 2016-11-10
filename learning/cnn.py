from __future__ import absolute_import
from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.advanced_activations import PReLU
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.utils import np_utils, generic_utils
from six.moves import range
from data import loadData
from keras import backend as K
K.set_image_dim_ordering('th')

data, label = loadData()
print(data.shape[0], ' samples')

label = np_utils.to_categorical(label, 10)

model = Sequential()

model.add(Convolution2D(4, 5, 5, border_mode = 'valid', input_shape = (1, 28, 28)))
model.add(Activation('tanh'))

model.add(Convolution2D(8, 3, 3, border_mode = 'valid'))
model.add(Activation('tanh'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Convolution2D(16, 3, 3, border_mode = 'valid'))
model.add(Activation('tanh'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('tanh'))

model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer="rmsprop")
model.fit(data, label, batch_size = 100, nb_epoch = 10, shuffle = True, verbose = 1, show_accuracy = True, validation_split = 0.2)
