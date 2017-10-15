import json
from keras import initializations
from keras.initializations import normal, identity
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD , Adam

IMAGE_WIDTH = 80
IMAGE_HEIGHT = 80
IMAGE_NUM_OF_CHANNELS = 4
NUM_OF_ACTIONS = 3 #number of valid actions: up, down or stay in place


class Player:
    """This class describes a player in the pong game.
        The player can be either the left or the right player.
        The player is represented as a neural network."""

    def __init__(self):
        self.model = None
        self.weights_file = None
        self.num_of_trains = 0
        self.num_of_wins_in_a_row = 0

    def build_model(self):
        print("Building the model")
        model = Sequential()
        model.add(
            Convolution2D(32, 8, 8, subsample=(4, 4), init=lambda shape, name: normal(shape, scale=0.01, name=name),
                          border_mode='same', input_shape=(IMAGE_NUM_OF_CHANNELS, IMAGE_WIDTH, IMAGE_HEIGHT)))
        model.add(Activation('relu'))
        model.add(
            Convolution2D(64, 4, 4, subsample=(2, 2), init=lambda shape, name: normal(shape, scale=0.01, name=name),
                          border_mode='same'))
        model.add(Activation('relu'))
        model.add(
            Convolution2D(64, 3, 3, subsample=(1, 1), init=lambda shape, name: normal(shape, scale=0.01, name=name),
                          border_mode='same'))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(512, init=lambda shape, name: normal(shape, scale=0.01, name=name)))
        model.add(Activation('relu'))
        model.add(Dense(NUM_OF_ACTIONS, init=lambda shape, name: normal(shape, scale=0.01, name=name)))

        adam = Adam(lr=1e-6)
        model.compile(loss='mse', optimizer=adam)
        print("The model was successfully built")
        self.model = model

    def load_model_weights(self, weights_file = None):
        if weights_file is not None:
            adam = Adam(lr=1e-6)
            self.model.load_weights(weights_file)
            self.model.compile(loss='mse', optimizer=adam)
            self.weights_file = weights_file



