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


test_all_command = "KERAS_BACKEND=theano THEANO_FLAGS=floatX=float32,device=gpu,force_device=True," \
                   "cuda.root=/usr/local/cuda,lib.cnmem=0.2 python ./test_all.py "

img_rows , img_cols = 80, 80
#Convert image into Black and white
img_channels = 4 #We stack 4 frames

def buildmodel():
    print("Now we build the model")
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
    print("We finish building the model")
    plot(model, to_file='model.png', show_shapes=True)
    return model


def trainNetwork(model,model2,args):
    player1_wins_in_a_row = 0
    player2_wins_in_a_row = 0

    player1_num_of_trains = 0
    player2_num_of_trains = 0

    learning_mode = (int(args['learning_mode']))

    if (learning_mode == 1):
        player1_num_of_trains = player1_num_of_trains + 1
    elif (learning_mode == 2):
        player2_num_of_trains = player2_num_of_trains + 1

    # open up a game state to communicate with emulator

    game_state = game.GameState()
    # store the previous observations in replay memory
    D2 = deque()

    # get the first state by doing nothing and preprocess the image to 80x80x4
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    x_t, r_0, terminal,_,_ = game_state.frame_step(do_nothing,do_nothing)

    x_t = skimage.color.rgb2gray(x_t)
    x_t = skimage.transform.resize(x_t,(80,80))
    x_t = skimage.exposure.rescale_intensity(x_t,out_range=(0,255))

    s_t = np.stack((x_t, x_t, x_t, x_t), axis=0)
    s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2])


    if args['mode'] == 'Run':
        print ("Run mode")

        OBSERVE = 999999999    #We keep observe, never train
        epsilon = FINAL_EPSILON
        print ("Now we load weight")
        model.load_weights("model1.h5")
        adam = Adam(lr=1e-6)
        model.compile(loss='mse',optimizer=adam)
        model2.load_weights("model2.h5")
        adam = Adam(lr=1e-6)
        model2.compile(loss='mse',optimizer=adam)
        print ("Weight load successfully")
        training_mode = False  # running
    else:                       #We go to training mode
        OBSERVE = OBSERVATION
        epsilon = INITIAL_EPSILON
        os.mkdir("trials" , 0755)
        learning_mode=int(args['learning_mode']) #which player learns


	# made changes 9.5:
	if os.path.isfile("model1.h5"): #check if file exists.
            model.load_weights("model1.h5")

	if os.path.isfile("model2.h5") : #check if file exists.
            model.load_weights("model2.h5")


        adam = Adam(lr=1e-6)
        model.compile(loss='mse', optimizer=adam)

        print("Weight load successfully")

        # printing log file
        training_mode = True # training

    observation_counter = 0
    num_folder=0
    start_time=datetime.datetime.now()

    while (True):
        loss = 0
        Q_sa = 0
        action_index = 0
        r_t = 0
        Q_sb = 0
        action_index2 = 0
        r_t2 = 0

        a_t = np.zeros([ACTIONS])
        a_t2 = np.zeros([ACTIONS])

        #choose an action epsilon greedy
        if observation_counter % FRAME_PER_ACTION == 0:

            q = model.predict(s_t)  # input a stack of 4 images, get the prediction
            max_Q = np.argmax(q)
            action_index = max_Q
            a_t[action_index] = 1

            if random.random() <= epsilon:  # for flayer1
                action_index2 = random.randrange(ACTIONS)
                a_t2[action_index2] = 1

            else:
                q = model2.predict(s_t)  # input a stack of 4 images, get the prediction
                max_Q = np.argmax(q)
                action_index2 = max_Q
                a_t2[action_index2] = 1


        #We reduced the epsilon gradually
        if epsilon > FINAL_EPSILON and observation_counter > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        #run the selected action and observed next state and reward
        if (learning_mode == 1):
            x_t1_colored, r_t, terminal, score,_ = game_state.frame_step(a_t2, a_t)

        elif (learning_mode == 2):
            x_t1_colored, r_t, terminal, score,_ = game_state.frame_step(a_t, a_t2)
            r_t = -r_t

        game_over=terminal

        x_t1_grey = skimage.color.rgb2gray(x_t1_colored)
        thresh = threshold_otsu(x_t1_grey)
        x_t1 = x_t1_grey > thresh  # binary image

        x_t1 = skimage.transform.resize(x_t1,(80,80))
        x_t1 = skimage.exposure.rescale_intensity(x_t1, out_range=(0, 255))
        x_t1 = x_t1.reshape(1, 1, x_t1.shape[0], x_t1.shape[1])

        s_t1 = np.append(x_t1, s_t[:, :3, :, :], axis=1)

        D2.append((s_t, action_index2, r_t, s_t1, terminal))
        if len(D2) > REPLAY_MEMORY:
            D2.popleft()

        #only train if done observing
        if observation_counter > OBSERVE:
            #sample a minibatch to train on
            minibatch = random.sample(D2, BATCH)

            inputs = np.zeros((BATCH, s_t.shape[1], s_t.shape[2], s_t.shape[3]))   #32, 80, 80, 4
            targets = np.zeros((inputs.shape[0], ACTIONS))                         #32, 2

            #Now we do the experience replay
            for i in range(0, len(minibatch)):
                state_t = minibatch[i][0]
                action_t = minibatch[i][1]   #This is action index
                reward_t = minibatch[i][2]
                state_t1 = minibatch[i][3]
                terminal = minibatch[i][4]
                # if terminated, only equals reward

                inputs[i:i + 1] = state_t    #I saved down s_t

                targets[i] = model2.predict(state_t)  # Hitting each buttom probability
                Q_sa = model2.predict(state_t1)

                if terminal:
                    targets[i, action_t] = reward_t
                else:
                    targets[i, action_t] = reward_t + GAMMA * np.max(Q_sa)

            loss += model2.train_on_batch(inputs, targets)

            # Now we do the experience replay

        s_t = s_t1
        observation_counter = observation_counter + 1

        # save progress every 10000 iterations
        if observation_counter % 100 == 0:
            #print("Now we save model")

            if learning_mode == 1:
                model.save_weights("model1.h5", overwrite=True)
                with open("model.json", "w") as outfile:
                    json.dump(model.to_json(), outfile)

            elif learning_mode == 2:
                model2.save_weights("model2.h5", overwrite=True)
                with open("model2.json", "w") as outfile:
                    json.dump(model2.to_json(), outfile)

        current_time=datetime.datetime.now()
        elapsedTime = (current_time - start_time).total_seconds()


        if(elapsedTime >= 30*60):
            num_folder+=1
            start_time=datetime.datetime.now()

            if learning_mode == 1:
                os.makedirs("trials/" + "player" + str(learning_mode) + "learning" +
                            str(player1_num_of_trains) + "/" + str(num_folder), 0755)

                shutil.copy2('model1.h5', "trials/" + "player" + str(learning_mode) + "learning" +
                            str(player1_num_of_trains) + "/" + str(num_folder) + '/model1.h5')

            else:
                os.makedirs("trials/" + "player" + str(learning_mode) + "learning" +
                         str(player2_num_of_trains) + "/" + str(num_folder), 0755)

                shutil.copy2('model2.h5', "trials/" + "player" + str(learning_mode) + "learning" +
                         str(player2_num_of_trains) + "/" + str(num_folder) + '/model2.h5')


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

            if (learning_mode == 1 and player1_wins_in_a_row == 50):
                subprocess.call('. ~/flappy/bin/activate && ' + test_all_command + " " +str(learning_mode) +
                                " " + str(player1_num_of_trains) + " " + str(player2_num_of_trains), shell=True)

                player1_wins_in_a_row = 0
                player2_wins_in_a_row = 0
                observation_counter = 0
                num_folder = 0
                learning_mode = 2
                player2_num_of_trains = player2_num_of_trains + 1
                start_time = datetime.datetime.now()

            elif (learning_mode == 2 and player2_wins_in_a_row == 50):
                subprocess.call('. ~/flappy/bin/activate && ' + test_all_command + " " + str(learning_mode) +
                                " " + str(player1_num_of_trains) + " " + str(player2_num_of_trains), shell=True)

                player1_wins_in_a_row = 0
                player2_wins_in_a_row = 0
                observation_counter = 0
                num_folder = 0
                learning_mode = 1
                player1_num_of_trains = player1_num_of_trains + 1
                start_time = datetime.datetime.now()


    print("Episode finished!")
    print("************************")




def playGame(args):
    model = buildmodel()
    model2 = buildmodel()
    trainNetwork(model,model2,args)

def main():
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-m','--mode', help='Train / Run', required=True)
    parser.add_argument('-l','--learning_mode', help='1,2', required=False)
    args = vars(parser.parse_args())
    playGame(args)

if __name__ == "__main__":
    main()

