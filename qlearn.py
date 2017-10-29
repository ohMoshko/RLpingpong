#!/usr/bin/env python

from __future__ import print_function
from player import Player
import matplotlib.pyplot as plt
import argparse
import skimage as skimage
from skimage import transform, color, exposure
from skimage.transform import rotate
from skimage.viewer import ImageViewer
from skimage.filters import threshold_otsu
import sys
import graphs

import random
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
from keras.optimizers import SGD, Adam
import pygame
from pygame.locals import *
from time import gmtime, strftime
import subprocess
import os
import datetime
import shutil
from keras.utils.visualize_util import plot
from aenum import Enum

IMAGE_WIDTH = 80
IMAGE_HEIGHT = 80
IMAGE_NUM_OF_CHANNELS = 4
NUM_OF_ACTIONS = 3  # number of valid actions: up, down or stay in place

GAME = 'pong'  # the name of the game being played for log files
GAMMA = 0.99  # decay rate of past observations
OBSERVATION = 320.  # timesteps to observe before training
EXPLORE = 3000000.  # frames over which to anneal gma
FINAL_EPSILON = 0.0001  # final value of epsilon
INITIAL_EPSILON = 0.1  # starting value of epsilon
REPLAY_MEMORY = 50000  # number of previous transitions to remember
BATCH = 32  # size of one minibatch
FRAME_PER_ACTION = 1

TEST_SEQUENTIAL_COMMAND = "KERAS_BACKEND=theano THEANO_FLAGS=floatX=float32,device=gpu,force_device=True," \
                          "cuda.root=/usr/local/cuda,lib.cnmem=0.2 python ./test_sequential_training.py "


class CurrentPlayer(Enum):
    left = 1
    right = 2


def copytree(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)


def num_of_lines_in_file(file_name):
    lines_num = 0
    with open(file_name) as f:
        size = sum(1 for _ in f)
    return lines_num


def save_weights_file(num_folder, current_training_player, left_player, right_player):
    num_folder += 1

    if current_training_player == CurrentPlayer.left:
        os.makedirs('trials_sequentially/' + 'left_player_learning' +
                    str(left_player.num_of_trains) + '/' + str(num_folder), 0755)

        shutil.copy2('./model1.h5', 'trials_sequentially/' + 'left_player_learning' +
                     str(left_player.num_of_trains) + '/' + str(num_folder) + '/model1.h5')

    elif current_training_player == CurrentPlayer.right:
        os.makedirs('trials_sequentially/' + 'right_player_learning' +
                    str(right_player.num_of_trains) + '/' + str(num_folder), 0755)

        shutil.copy2('./model2.h5', 'trials_sequentially/' + 'right_player_learning' +
                     str(right_player.num_of_trains) + '/' + str(num_folder) + '/model2.h5')

    return num_folder


def train_sequentially(left_player, right_player, first_learning_player):
    if os.path.exists('logs'):
        if os.path.exists('old_logs'):
            shutil.rmtree('old_logs')
        os.mkdir('old_logs', 0755)
        copytree('logs', 'old_logs')
        shutil.rmtree('logs')
    os.mkdir('logs', 0755)

    # moving old trials to old_trials folder
    # TODO: remove comment from trials folder delete and make!!!!!!
    #if os.path.exists('trials_sequentially'):
    #    if os.path.exists('old_trials_sequentially'):
    #        shutil.rmtree('old_trials_sequentially')
    #    os.mkdir('old_trials_sequentially', 0755)
    #    copytree('trials_sequentially', 'old_trials_sequentially')
    #    shutil.rmtree('trials_sequentially')
    #os.mkdir('trials_sequentially', 0755)

    current_log_folder = ''
    # TODO: delete!!!!!!
    left_player.num_of_trains = 1
    if (first_learning_player == CurrentPlayer.left):
        current_training_player = CurrentPlayer.left
        left_player.num_of_trains += 1
        current_log_folder = 'logs/' + 'left_player' + \
                             '_learning' + str(left_player.num_of_trains)

    elif (first_learning_player == CurrentPlayer.right):
        current_training_player = CurrentPlayer.right
        right_player.num_of_trains += 1
        current_log_folder = 'logs/' + 'right_player' + \
                             '_learning' + str(right_player.num_of_trains)

    # open up a game state to communicate with emulator
    game_state = game.GameState()

    # store the previous observations in replay memory
    D1 = deque()
    D2 = deque()

    # get the first state (do nothing) and pre-process the image to 4x80x80
    do_nothing = np.zeros(NUM_OF_ACTIONS)
    do_nothing[0] = 1

    single_game_frame, r_0, terminal, _, _ = game_state.frame_step(do_nothing, do_nothing)

    single_game_frame = skimage.color.rgb2gray(single_game_frame)
    single_game_frame = skimage.transform.resize(single_game_frame, (IMAGE_WIDTH, IMAGE_HEIGHT))
    single_game_frame = skimage.exposure.rescale_intensity(single_game_frame, out_range=(0, 255))

    # stacking 4 images together to form a state: 4 images = state
    current_state = np.stack((single_game_frame, single_game_frame, single_game_frame,
                              single_game_frame), axis=0)

    current_state = current_state.reshape(1, current_state.shape[0],
                                          current_state.shape[1],
                                          current_state.shape[2])

    epsilon = INITIAL_EPSILON
    observation_counter = 0
    num_folder = 0
    start_time = datetime.datetime.now()
    start_time_loss = datetime.datetime.now()
    losses = []
    episode_number = 0

    while True:
        loss = 0
        Q_sa = 0

        # player1 - left player
        action_index1 = 0
        reward1 = 0
        action_left_player = np.zeros([NUM_OF_ACTIONS])

        # player2 - right player
        action_index2 = 0
        reward2 = 0
        action_right_player = np.zeros([NUM_OF_ACTIONS])

        exploration_counter = 0
        exploration_flag = 0
        # choose an action epsilon greedy
        if (observation_counter % FRAME_PER_ACTION) == 0:
            if current_training_player == CurrentPlayer.left:
                #q = left_player.model.predict(current_state)  # input a stack of 4 images, get the prediction
                #max_Q = np.argmax(q)
                #moves_options = [0, 1, 2]
                #moves_options.remove(max_Q)
                #if random.random() <= epsilon or exploration_flag == 1:
                if random.random() <= epsilon:
                    #q = left_player.model.predict(current_state)  # input a stack of 4 images, get the prediction
                    #max_Q = np.argmax(q)
                    #exploration_flag = 1
                    #exploration_counter +=1
                    action_index1 = random.randrange(NUM_OF_ACTIONS)
                    #action_index1 = random.choice(moves_options)
                    action_left_player[action_index1] = 1
                    #if exploration_counter > 10:
                    #    exploration_flag = 0
                    #    exploration_counter = 0
                else:
                    q = left_player.model.predict(current_state)  # input a stack of 4 images, get the prediction
                    max_Q = np.argmax(q)
                    action_index1 = max_Q
                    action_left_player[action_index1] = 1

                # right player:
                q = right_player.model.predict(current_state)  # input a stack of 4 images, get the prediction
                max_Q = np.argmax(q)
                action_index2 = max_Q
                action_right_player[action_index2] = 1

            elif current_training_player == CurrentPlayer.right:
                if random.random() <= epsilon:
                    action_index2 = random.randrange(NUM_OF_ACTIONS)
                    action_right_player[action_index2] = 1

                else:
                    q = right_player.model.predict(current_state)  # input a stack of 4 images, get the prediction
                    max_Q = np.argmax(q)
                    action_index2 = max_Q
                    action_right_player[action_index2] = 1

                # left player:
                q = left_player.model.predict(current_state)  # input a stack of 4 images, get the prediction
                max_Q = np.argmax(q)
                action_index1 = max_Q
                action_left_player[action_index1] = 1

        # we reduced the epsilon gradually
        if (epsilon > FINAL_EPSILON) and (observation_counter > OBSERVATION):
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        single_game_frame_colored, reward1, terminal, score, _ = \
            game_state.frame_step(action_left_player, action_right_player)

        if current_training_player == CurrentPlayer.right:
            reward2 = -reward1

        single_game_frame_grey = skimage.color.rgb2gray(single_game_frame_colored)
        thresh = threshold_otsu(single_game_frame_grey)
        single_game_frame = single_game_frame_grey > thresh

        single_game_frame = skimage.transform.resize(single_game_frame, (80, 80))
        single_game_frame = skimage.exposure.rescale_intensity(single_game_frame, out_range=(0, 255))
        single_game_frame = single_game_frame.reshape(1, 1, single_game_frame.shape[0],
                                                                    single_game_frame.shape[1])
        # next 4 images = next state
        next_state = np.append(single_game_frame, current_state[:, :3, :, :], axis=1)

        if (current_training_player == CurrentPlayer.left):
            D1.append((current_state, action_index1, reward1, next_state, terminal))
            if len(D1) > REPLAY_MEMORY:
                D1.popleft()

        elif (current_training_player == CurrentPlayer.right):
            D2.append((current_state, action_index2, reward2, next_state, terminal))
            if len(D2) > REPLAY_MEMORY:
                D2.popleft()

        # only train if done observing
        if observation_counter > OBSERVATION:
            # sample a minibatch to train on - eliminates states correlation
            if current_training_player == CurrentPlayer.left:
                minibatch = random.sample(D1, BATCH)
            elif current_training_player == CurrentPlayer.right:
                minibatch = random.sample(D2, BATCH)

            inputs = np.zeros((BATCH, current_state.shape[1], current_state.shape[2],
                               current_state.shape[3]))  # 32, 4, 80, 80
            targets = np.zeros((inputs.shape[0], NUM_OF_ACTIONS))  # 32, 2

            # experience replay
            for i in range(0, len(minibatch)):
                current_state_t = minibatch[i][0]
                action_t = minibatch[i][1]  # This is action index
                reward_t = minibatch[i][2]
                next_state_t = minibatch[i][3]
                terminal_t = minibatch[i][4]

                inputs[i:i + 1] = current_state_t

                if (current_training_player == CurrentPlayer.left):
                    targets[i] = left_player.model.predict(current_state_t)
                    Q_sa = left_player.model.predict(next_state_t)
                elif (current_training_player == CurrentPlayer.right):
                    targets[i] = right_player.model.predict(current_state_t)
                    Q_sa = right_player.model.predict(next_state_t)

                if terminal_t:
                    targets[i, action_t] = reward_t
                else:
                    targets[i, action_t] = reward_t + GAMMA * np.max(Q_sa)

            if (current_training_player == CurrentPlayer.left):
                loss += left_player.model.train_on_batch(inputs, targets)
            elif (current_training_player == CurrentPlayer.right):
                loss += right_player.model.train_on_batch(inputs, targets)

            # log_file_loss_path = current_log_folder + '/loss'
            # log_file_qmax_path = current_log_folder + '/qmax'
            # num_of_lines_in_loss_file = 0
            # loss_file = None
            #
            # if not os.path.exists(current_log_folder):
            #     os.mkdir(current_log_folder, 0755)
            #
            # elapsed_time_loss = (current_time - start_time_loss).total_seconds()
            #
            # if elapsed_time_loss >= 1 * 60:
            #     episode_number += 1
            #     start_time_loss = datetime.datetime.now()
            #     loss_file = open(log_file_loss_path, 'a')
            #     num_of_lines_in_loss_file = num_of_lines_in_file(log_file_loss_path)
            #
            #     #with open(log_file_loss_path, 'a') as loss_file:
            #     loss_file.write(str(episode_number) + ' : ' + str(loss) + '\n')
            #     loss_file.flush()
            #     loss_file.close()

        current_state = next_state
        observation_counter = observation_counter + 1

        # save progress (updating weights files) every 100 iterations
        if observation_counter % 100 == 0:
            if current_training_player == CurrentPlayer.left:
                left_player.model.save_weights('model1.h5', overwrite=True)
                with open('model1.json', 'w') as outfile:
                    json.dump(left_player.model.to_json(), outfile)

            elif current_training_player == CurrentPlayer.right:
                right_player.model.save_weights('model2.h5', overwrite=True)
                with open('model2.json', 'w') as outfile:
                    json.dump(right_player.model.to_json(), outfile)

        current_time = datetime.datetime.now()
        elapsed_time = (current_time - start_time).total_seconds()

        if elapsed_time >= 5 * 60:
            start_time = datetime.datetime.now()
            num_folder = save_weights_file(num_folder, current_training_player,
                                           left_player, right_player)

        if terminal:  # game over
            if score[0] < score[1]:
                right_player.num_of_wins_in_a_row += 1
                left_player.num_of_wins_in_a_row = 0
            elif score[1] < score[0]:
                left_player.num_of_wins_in_a_row += 1
                right_player.num_of_wins_in_a_row = 0

            if (current_training_player == CurrentPlayer.left and left_player.num_of_wins_in_a_row == 20):
                _ = save_weights_file(num_folder, current_training_player, left_player, right_player)
                # graphs.plot_loss(current_log_folder, current_log_folder + '/loss')
                #subprocess.call('. ~/flappy/bin/activate && ' + TEST_SEQUENTIAL_COMMAND + ' ' + str(1) +
                #                 ' ' + str(left_player.num_of_trains) + ' ' + str(right_player.num_of_trains),
                #                 shell=True)

                left_player.num_of_wins_in_a_row = 0
                right_player.num_of_wins_in_a_row = 0
                observation_counter = 0
                epsilon = INITIAL_EPSILON
                num_folder = 0
                current_training_player = CurrentPlayer.right
                right_player.num_of_trains += 1
                current_log_folder = 'logs/' + 'right_player' + \
                                     '_learning' + str(right_player.num_of_trains)
                episode_number = 0
                D1.clear()
                start_time = datetime.datetime.now()
                #break

            elif (current_training_player == CurrentPlayer.right and right_player.num_of_wins_in_a_row == 20):
                # graphs.plot_loss(current_log_folder, current_log_folder + '/loss')
                _ = save_weights_file(num_folder, current_training_player, left_player, right_player)
                #subprocess.call('. ~/flappy/bin/activate && ' + TEST_SEQUENTIAL_COMMAND + ' ' + str(2) +
                #                 ' ' + str(left_player.num_of_trains) + ' ' + str(right_player.num_of_trains),
                #                 shell=True)

                left_player.num_of_wins_in_a_row = 0
                right_player.num_of_wins_in_a_row = 0
                observation_counter = 0
                epsilon = INITIAL_EPSILON
                num_folder = 0
                current_training_player = CurrentPlayer.left
                left_player.num_of_trains += 1
                current_log_folder = 'logs/' + 'left_player' + \
                                     '_learning' + str(left_player.num_of_trains)
                episode_number = 0
                D2.clear()
                start_time = datetime.datetime.now()
                #break

    print("Episode finished!")
    print("************************")


def play_game(args):
    if not os.path.isfile('model1.h5') or not os.path.isfile('model2.h5'):
        print("Weights files are missing!")

    left_player = Player()
    left_player.build_model()
    left_player.load_model_weights('model1.h5')
    right_player = Player()
    right_player.build_model()
    right_player.load_model_weights('model2.h5')

    if int(args['first_learning_player']) == 1:
        first_learning_player = CurrentPlayer.left
    elif int(args['first_learning_player']) == 2:
        first_learning_player = CurrentPlayer.right

    if args['learning_mode'] == 'SEQUENTIALLY':  # one after another (left and then right and then left...)
        train_sequentially(left_player, right_player, first_learning_player)
        # elif (args['learning_mode'] == 'SIMULTANEOUSLY'):
        #     train_simultaneously(left_player, right_player)


def main():
    parser = argparse.ArgumentParser(description='Ping-Pong Q-Learning')
    parser.add_argument('-l', '--learning_mode', help='SEQUENTIALLY, SIMULTANEOUSLY', required=True)
    parser.add_argument('-p', '--first_learning_player', help='1, 2', required=False)
    args = vars(parser.parse_args())
    play_game(args)


if __name__ == "__main__":
    main()
