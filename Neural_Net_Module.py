#build the NN models: RNN module
import tensorflow

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten,Conv1D,MaxPooling1D,BatchNormalization

def dnn_model(input_dim):
    model = Sequential()

    model.add(Conv1D(32, 3, strides=1, activation='relu', input_shape=(640, 1)))
    model.add(Conv1D(32, 3, strides=1, activation='relu'))
    model.add(MaxPooling1D(3, 1))
    model.add(Activation('relu'))
    model.add(BatchNormalization())


    model.add(Conv1D(64, 3, strides=1, activation='relu'))
    model.add(Conv1D(64, 3,strides=1, activation='relu'))
    model.add(MaxPooling1D(3, 1))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Conv1D(128, 3,strides=1, activation='relu'))
    model.add(Conv1D(128, 3, strides=1, activation='relu'))
    model.add(MaxPooling1D(3, 1))
    model.add(Activation('relu'))
    model.add(BatchNormalization())


    model.add(Flatten())


    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(128, activation='relu', name = "visualize"))
    model.add(Dense(17))
    model.add(Activation('softmax'))

    return model