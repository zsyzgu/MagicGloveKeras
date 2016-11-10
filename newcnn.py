from __future__ import absolute_import
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.advanced_activations import PReLU
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.utils import np_utils, generic_utils
from data_loader import load_data

#from __future__ import print_function
from six.moves import range

from keras import backend as K
K.set_image_dim_ordering('th')

data,out = load_data();
my_model = Sequential()
my_model.add(Convolution2D(4, 5, 5, border_mode='valid',input_shape=(2,37,65))) 
my_model.add(Activation('tanh'))
my_model.add(MaxPooling2D(pool_size=(2, 2)))

my_model.add(Convolution2D(8, 3, 3, border_mode='valid'))
my_model.add(Activation('tanh'))
my_model.add(MaxPooling2D(pool_size=(2, 2)))

my_model.add(Flatten())
my_model.add(Dense(128))
my_model.add(Activation('softmax'))

my_model.add(Dense(63))
my_model.add(Activation('softmax'))

my_model.compile(loss='categorical_crossentropy', optimizer="rmsprop")

my_model.fit(data, out, batch_size=100,nb_epoch=1000,shuffle=True,verbose=1,show_accuracy=True,validation_split=0.2)

#out = my_model.predict(data[0:1], batch_size=1)
#print out
