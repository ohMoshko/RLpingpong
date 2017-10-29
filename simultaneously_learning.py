#!/usr/bin/env python
from __future__ import print_function

import matplotlib.pyplot as plt
import argparse
import skimage as skimage
from skimage import transform, color, exposure
from skimage.transform import rotate
from skimage.viewer import ImageViewer
from skimage.filters import threshold_otsu
import sys
sys.path.append("game/")
import pong_fun as game
import random
import numpy as np
from collections import deque

import json
from keras import initializations
from keras.initializations import normal, identity
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD , Adam


import pygame
from pygame.locals import *
from time import gmtime, strftime

import subprocess
import os
import datetime
import shutil

from keras.utils.visualize_util import plot
counter=1
GAME = 'pong' # the name of the game being played for log files
CONFIG = 'nothreshold'
ACTIONS = 3 # number of valid actions
GAMMA = 0.99 # decay rate of past observations
OBSERVATION = 320. # timesteps to observe before training
EXPLORE = 3000000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001 # final value of epsilon
INITIAL_EPSILON = 0.1 # starting value of epsilon
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
FRAME_PER_ACTION = 1
IMAGE_WIDTH = 80 #screen's width
IMAGE_HEIGHT = 80 #screen's height
IMAGE_DEPTH = 4

test_all_command = "KERAS_BACKEND=theano THEANO_FLAGS=floatX=float32,device=gpu,force_device=True," \
                   "cuda.root=/usr/local/cuda,lib.cnmem=0.2 python ./test_all.py "

img_rows , img_cols = 80, 80
#Convert image into Black and white
img_channels = 4 #We stack 4 frames

def copytree(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)

def buildmodel():
    print("Building the model")
    model = Sequential()
    model.add(Convolution2D(32, 8, 8, subsample=(4,4),init=lambda shape, name: normal(shape, scale=0.01, name=name), border_mode='same',input_shape=(img_channels,img_rows,img_cols)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 4, 4, subsample=(2,2),init=lambda shape, name: normal(shape, scale=0.01, name=name), border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, subsample=(1,1),init=lambda shape, name: normal(shape, scale=0.01, name=name), border_mode='same'))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512, init=lambda shape, name: normal(shape, scale=0.01, name=name)))
    model.add(Activation('relu'))
    model.add(Dense(ACTIONS,init=lambda shape, name: normal(shape, scale=0.01, name=name)))

    adam = Adam(lr=1e-6)
    model.compile(loss='mse',optimizer=adam)
    #model.summary()
    print("Model was successfully built")
    plot(model, to_file='model.png', show_shapes=True)
    return model


def trainNetwork(model1,model2,args):
    player1_wins_in_a_row = 0
    player2_wins_in_a_row = 0

    player1_num_of_trains = 0
    player2_num_of_trains = 0

    # open up a game state to communicate with emulator
    game_state = game.GameState()

    # store the previous observations in replay memory
    D1 = deque()
    D2 = deque()

    # get the first state by doing nothing and preprocess the image to 80x80x4
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    x_t1, r_0, terminal, _, _ = game_state.frame_step(do_nothing,do_nothing)

    x_t1 = skimage.color.rgb2gray(x_t1)
    x_t1 = skimage.transform.resize(x_t1,(80,80))
    x_t1 = skimage.exposure.rescale_intensity(x_t1,out_range=(0,255))

    x_t2 = np.flipud(x_t1)

    player1_curr_state = np.stack((x_t1, x_t1, x_t1, x_t1), axis=0)
    player1_curr_state = player1_curr_state.reshape(1, player1_curr_state.shape[0],
                                            player1_curr_state.shape[1], player1_curr_state.shape[2])

    player2_curr_state = np.stack((x_t2, x_t2, x_t2, x_t2), axis=0)
    player2_curr_state = player2_curr_state.reshape(1, player2_curr_state.shape[0],
                                            player2_curr_state.shape[1], player2_curr_state.shape[2])


    #training mode
    OBSERVE = OBSERVATION
    epsilon = INITIAL_EPSILON

    #moving old trials to old_trials folder
    if os.path.exists("trials_simultaneously"):
        copytree("trials_simultaneously", "old_trials_simultaneously")
        shutil.rmtree("trials_simultaneously")

    os.mkdir("trials_simultaneously" , 0755)
    learning_mode=int(args['learning_mode']) #which player learns


    if os.path.isfile("model1.h5"):
        model1.load_weights("model1.h5")

    if os.path.isfile("model2.h5"):
        model2.load_weights("model2.h5")


    adam = Adam(lr=1e-6)
    model1.compile(loss='mse', optimizer=adam)
    model2.compile(loss='mse', optimizer=adam)

    print("Weights loaded successfully")

    training_mode = True # training

    observation_counter = 0
    num_folder=0
    start_time=datetime.datetime.now()

    observation_counter = 0

    while (True):
        loss1 = 0
        loss2 = 0
        Q_sa1 = 0
        action_index1 = 0
        r_t1 = 0
        loss2 = 0
        Q_sa2 = 0
        action_index2 = 0
        r_t2 = 0

        a_t1 = np.zeros([ACTIONS])
        a_t2 = np.zeros([ACTIONS])

        #choose an action epsilon greedy
        if (observation_counter % FRAME_PER_ACTION) == 0:

            if random.random() <= epsilon:  # for flayer1
                action_index1 = random.randrange(ACTIONS)
                a_t1[action_index1] = 1

            else:
                q = model1.predict(player1_curr_state)  # input a stack of 4 images, get the prediction
                max_Q = np.argmax(q)
                action_index1 = max_Q
                a_t1[action_index1] = 1

            if random.random() <= epsilon:  # for flayer2
                action_index2 = random.randrange(ACTIONS)
                a_t2[action_index2] = 1

            else:
                q = model2.predict(player2_curr_state)  # input a stack of 4 images, get the prediction
                max_Q = np.argmax(q)
                action_index2 = max_Q
                a_t2[action_index2] = 1


        #We reduced the epsilon gradually
        if (epsilon > FINAL_EPSILON) and (observation_counter > OBSERVE):
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE


        # run the selected action and observed next state and reward
        x_t1_colored, r_t1, terminal, score, _ = game_state.frame_step(a_t1, a_t2)
        r_t2 = r_t1 * (-1)
        game_over = terminal

        x_t1_grey = skimage.color.rgb2gray(x_t1_colored)
        thresh = threshold_otsu(x_t1_grey)
        x_t1 = x_t1_grey > thresh  # binary image
        x_t2 = np.flipud(x_t1)

        x_t1 = skimage.transform.resize(x_t1, (80, 80))
        x_t1 = skimage.exposure.rescale_intensity(x_t1, out_range=(0, 255))

        x_t2 = skimage.transform.resize(x_t2, (80, 80))
        x_t2 = skimage.exposure.rescale_intensity(x_t2, out_range=(0, 255))

        x_t1 = x_t1.reshape(1, 1, x_t1.shape[0], x_t1.shape[1])
        x_t2 = x_t2.reshape(1, 1, x_t2.shape[0], x_t2.shape[1])

        player1_next_state = np.append(x_t1, player1_curr_state[:, :3, :, :], axis=1)
        player2_next_state = np.append(x_t2, player2_curr_state[:, :3, :, :], axis=1)

        # store the transition in D
        D1.append((player1_curr_state, action_index1, r_t1, player1_next_state, terminal))
        if len(D1) > REPLAY_MEMORY:
            D1.popleft()

        D2.append((player2_curr_state, action_index2, r_t2, player2_next_state, terminal))
        if len(D2) > REPLAY_MEMORY:
            D2.popleft()

        if observation_counter > OBSERVE:
            # sample a minibatch to train on
            minibatch1 = random.sample(D1, BATCH)
            minibatch2 = random.sample(D2, BATCH)

            inputs1 = np.zeros((BATCH, IMAGE_DEPTH, IMAGE_WIDTH, IMAGE_HEIGHT))  # 32, 4, 80, 80
            targets1 = np.zeros((BATCH, ACTIONS))  # 32, 2

            inputs2 = np.zeros((BATCH, IMAGE_DEPTH, IMAGE_WIDTH, IMAGE_HEIGHT))  # 32, 4, 80, 80
            targets2 = np.zeros((BATCH, ACTIONS))  # 32, 2

            # Now we do the experience replay
            for i in range(0, len(minibatch1)):
                curr_state_t1 = minibatch1[i][0]
                action_t1 = minibatch1[i][1]  # This is action index
                reward_t1 = minibatch1[i][2]
                next_state_t1 = minibatch1[i][3]
                terminal1 = minibatch1[i][4]
                # if terminated, only equals reward

                inputs1[i:i + 1] = curr_state_t1  # I saved down s_t1

                targets1[i] = model1.predict(curr_state_t1)  # Hitting each buttom probability
                Q_sa1 = model1.predict(curr_state_t1)

                if terminal1:
                    targets1[i, action_t1] = reward_t1
                else:
                    targets1[i, action_t1] = reward_t1 + GAMMA * np.max(Q_sa1)

            loss1 += model1.train_on_batch(inputs1, targets1)

            # Now we do the experience replay
            for i in range(0, len(minibatch2)):
                curr_state_t2 = minibatch2[i][0]
                action_t2 = minibatch2[i][1]  # This is action index
                reward_t2 = minibatch2[i][2]
                next_state_t2 = minibatch2[i][3]
                terminal2 = minibatch2[i][4]
                # if terminated, only equals reward

                inputs2[i:i + 1] = curr_state_t2

                targets2[i] = model2.predict(curr_state_t2)  # Hitting each buttom probability
                Q_sa2 = model2.predict(curr_state_t2)

                if terminal2:
                    targets2[i, action_t2] = reward_t2
                else:
                    targets2[i, action_t2] = reward_t2 + GAMMA * np.max(Q_sa2)


            loss2 += model2.train_on_batch(inputs2, targets2)

        player1_curr_state = player1_next_state
        player2_curr_state = player2_next_state
        observation_counter = observation_counter + 1


        # save progress every 10000 iterations
        if observation_counter % 100 == 0:
            #print("Now we save model")

            if learning_mode == 1:
                model1.save_weights("model1.h5", overwrite=True)
                with open("model1.json", "w") as outfile1:
                    json.dump(model1.to_json(), outfile1)

            elif learning_mode == 2:
                model2.save_weights("model2.h5", overwrite=True)
                with open("model2.json", "w") as outfile2:
                    json.dump(model2.to_json(), outfile2)

        current_time=datetime.datetime.now()
        elapsedTime = (current_time - start_time).total_seconds()


        if(elapsedTime >= 30*60):
            num_folder+=1
            start_time=datetime.datetime.now()

            os.makedirs("trials_simultaneously/" + "player" + str(1) + "learning" +
                        "/" + str(num_folder), 0755)

            shutil.copy2('model1.h5', "trials_simultaneously/" + "player" + str(1) + "learning" +
                        "/" + str(num_folder) + '/model1.h5')

            os.makedirs("trials_simultaneously/" + "player" + str(2) + "learning" +
                        "/" + str(num_folder), 0755)

            shutil.copy2('model2.h5', "trials_simultaneously/" + "player" + str(2) + "learning" +
                        "/" + str(num_folder) + '/model2.h5')


        if (game_over) :
            if score[0] < score[1]:
                player2_wins_in_a_row = player2_wins_in_a_row + 1
                player1_wins_in_a_row = 0
                percentage = 0.0
            elif score[1] < score[0]:
                player1_wins_in_a_row = player1_wins_in_a_row + 1
                player2_wins_in_a_row = 0
                percentage = 1.0
            else:
                percentage = (score[0] / float((score[0] + score[1])))


    print("Episode finished!")
    print("************************")




def playGame(args):
    model1 = buildmodel()
    model2 = buildmodel()
    trainNetwork(model1,model2,args)

def main():
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-m','--mode', help='Train / Run', required=True)
    parser.add_argument('-l','--learning_mode', help='1,2', required=False)
    args = vars(parser.parse_args())
    playGame(args)

if __name__ == "__main__":
    main()

