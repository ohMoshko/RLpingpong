#!/usr/bin/env python
from __future__ import print_function

import matplotlib.pyplot as plt
import argparse
import skimage as skimage
from skimage import transform, color, exposure
from skimage.transform import rotate
from skimage.viewer import ImageViewer
import sys

sys.path.append("game/")
import pong_fun as game
import random
import numpy as np
from collections import deque
import datetime
import json
from keras import initializations
from keras.initializations import normal, identity
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adam

import pygame
from pygame.locals import *
from time import gmtime, strftime
import os.path

counter = 1
GAME = 'snake'  # the name of the game being played for log files
CONFIG = 'nothreshold'
ACTIONS = 3  # number of valid actions
GAMMA = 0.99  # decay rate of past observations
OBSERVATION = 10000000.  # timesteps to observe before training
EXPLORE = 12000.  # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001  # final value of epsilon
INITIAL_EPSILON = 0.1  # starting value of epsilon
REPLAY_MEMORY = 50000  # number of previous transitions to remember
BATCH = 32  # size of minibatch
FRAME_PER_ACTION = 1

img_rows, img_cols = 80, 80
# Convert image into Black and white
img_channels = 4  # We stack 4 frames


def buildmodel():
    print("Now we build the model")
    model = Sequential()
    model.add(Convolution2D(32, 8, 8, subsample=(4, 4), init=lambda shape, name: normal(shape, scale=0.01, name=name),
                            border_mode='same', input_shape=(img_channels, img_rows, img_cols)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 4, 4, subsample=(2, 2), init=lambda shape, name: normal(shape, scale=0.01, name=name),
                            border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), init=lambda shape, name: normal(shape, scale=0.01, name=name),
                            border_mode='same'))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512, init=lambda shape, name: normal(shape, scale=0.01, name=name)))
    model.add(Activation('relu'))
    model.add(Dense(ACTIONS, init=lambda shape, name: normal(shape, scale=0.01, name=name)))
    adam = Adam(lr=1e-6)
    model.compile(loss='mse', optimizer=adam)
    print("We finish building the model")
    return model


def trainNetwork(model, args):
    # open up a game state to communicate with emulator

    game_state = game.GameState()
    # store the previous observations in replay memory
    D = deque()

    # get the first state by doing nothing and preprocess the image to 80x80x4
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    x_t, r_0, terminal, _ = game_state.frame_step(do_nothing)

    x_t = skimage.color.rgb2gray(x_t)
    x_t = skimage.transform.resize(x_t, (80, 80))
    x_t = skimage.exposure.rescale_intensity(x_t, out_range=(0, 255))

    s_t = np.stack((x_t, x_t, x_t, x_t), axis=0)

    # In Keras, need to reshape
    s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2])

    learning_mode = 0  # 2 for learng based on human, 3 for reverse reinforcement
    if args['mode'] == 'Run':
        OBSERVE = 999999999  # We keep observe, never train
        epsilon = FINAL_EPSILON
        print("Now we load weight")
        model.load_weights("model1.h5")
        adam = Adam(lr=1e-6)
        model.compile(loss='mse', optimizer=adam)
        print("Weight load successfully")
        training_mode = False  # running
        os.mkdir("pic", 0755);
        a_file = open("logs_" + GAME + "/logfile_" + str(counter) + ".txt", 'a')
    else:  # We go to training mode
        OBSERVE = OBSERVATION
        epsilon = INITIAL_EPSILON
        learning_mode = int(args['learning_mode'])

        if os.path.isfile("model.h5"):  # check if file exists.
            model.load_weights("model.h5")
            adam = Adam(lr=1e-6)
            model.compile(loss='mse', optimizer=adam)
            print("Weight load successfully")

        # printing log file
        training_mode = True  # training

    j = 1
    os.mkdir("pic/" + str(j), 0755);
    t = 0
    pic_counter = 0

    while (True):
        loss = 0
        Q_sa = 0
        action_index = 0
        r_t = 0
        a_t = np.zeros([ACTIONS])

        # choose an action epsilon greedy
        if t % FRAME_PER_ACTION == 0:
            if not training_mode:  # running
                q = model.predict(s_t)  # input a stack of 4 images, get the prediction
                max_Q = np.argmax(q)
                action_index = max_Q
                a_t[action_index] = 1

        # We reduced the epsilon gradually
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        # run the selected action and observed next state and reward
        x_t1_colored, r_t, terminal, score = game_state.frame_step(a_t)

        game_over = terminal

        x_t1 = skimage.color.rgb2gray(x_t1_colored)

        if (score >= 0):
            fig1 = plt.figure(pic_counter)
            plt.imshow(x_t1_colored)
            print('time now: ', datetime.datetime.now())
        fig1.savefig('pic/' + str(j) + '/' + str(pic_counter) + 'colored pic.png')

        plt.close()

        x_t1 = skimage.transform.resize(x_t1, (80, 80))
        x_t1 = skimage.exposure.rescale_intensity(x_t1, out_range=(0, 255))
        x_t1 = x_t1.reshape(1, 1, x_t1.shape[0], x_t1.shape[1])
        s_t1 = np.append(x_t1, s_t[:, :3, :, :], axis=1)

        # store the transition in D
        D.append((s_t, action_index, r_t, s_t1, terminal))
        if len(D) > REPLAY_MEMORY:
            D.popleft()

        s_t = s_t1
        t = t + 1
        pic_counter += 1

        # print info
        state = ""
        if t <= OBSERVE:
            state = "observe"
        elif t > OBSERVE and t <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        if (game_over):
            j = j + 1
            os.mkdir("pic/" + str(j), 0755);
            print(j, "score: ", score, file=a_file)
            a_file.flush()
            pic_counter = 0



    print("Episode finished!")
    print("************************")


def playGame(args):
    model = buildmodel()
    trainNetwork(model, args)


def main():
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-m', '--mode', help='Train / Run', required=True)
    parser.add_argument('-l', '--learning_mode', help='1,2,3,4', required=False)
    args = vars(parser.parse_args())
    playGame(args)


if __name__ == "__main__":
    main()
