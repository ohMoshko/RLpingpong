#!/usr/bin/env python

from __future__ import print_function
import argparse
import sys
import skimage as skimage
from skimage import transform, color, exposure
from skimage.filters import threshold_otsu

from player import Player

import matplotlib.pyplot as plt
sys.path.append("game/")
import pong_fun as game
import pong_for_simultaneously_training as game_sim_training
import random
import numpy as np
from collections import deque
import json
import os
import datetime
import shutil
from file_utils import copytree, save_weights_file

IMAGE_WIDTH = 80
IMAGE_HEIGHT = 80
IMAGE_NUM_OF_CHANNELS = 4
NUM_OF_ACTIONS = 3  # number of valid actions: up, down or stay in place

GAME = 'pong'  # the name of the game being played for log files
GAMMA = 0.99  # decay rate of past observations
OBSERVATION = 320.  # timesteps to observe before training
EXPLORE = 3000000.  # frames over which to anneal gma
FINAL_EPSILON = 0.0001  # final value of epsilon
# FINAL_EPSILON = 0.0005  # for the second learning
INITIAL_EPSILON = 0.1  # starting value of epsilon
REPLAY_MEMORY = 50000  # number of previous transitions to remember
BATCH = 32  # size of one minibatch
FRAME_PER_ACTION = 1

TEST_SEQUENTIAL_COMMAND = "KERAS_BACKEND=theano THEANO_FLAGS=floatX=float32,device=gpu,force_device=True," \
                          "cuda.root=/usr/local/cuda,lib.cnmem=0.2 python ./test_sequential_training.py "


class CurrentPlayer:
    def __init__(self):
        pass

    Left, Right, Both = range(1, 4)


def train_sequentially(left_player, right_player, first_learning_player):
    if os.path.exists('logs'):
        if os.path.exists('old_logs'):
            shutil.rmtree('old_logs')
        os.mkdir('old_logs', 0755)
        copytree('logs', 'old_logs')
        shutil.rmtree('logs')
    os.mkdir('logs', 0755)

    # moving old trials to old_trials folder
    if os.path.exists('trials_sequentially'):
        if os.path.exists('old_trials_sequentially'):
            shutil.rmtree('old_trials_sequentially')
        os.mkdir('old_trials_sequentially', 0755)
        copytree('trials_sequentially', 'old_trials_sequentially')
        shutil.rmtree('trials_sequentially')
    os.mkdir('trials_sequentially', 0755)

    current_log_folder = ''
    left_player.num_of_trains = 1
    if (first_learning_player == CurrentPlayer.Left):
        current_training_player = CurrentPlayer.Left
        left_player.num_of_trains += 1
        current_log_folder = 'logs/' + 'left_player' + \
                             '_learning' + str(left_player.num_of_trains)

    elif (first_learning_player == CurrentPlayer.Right):
        current_training_player = CurrentPlayer.Right
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

    single_game_frame, _, _, terminal, _, _ = game_state.frame_step(do_nothing, do_nothing)

    single_game_frame = skimage.color.rgb2gray(single_game_frame)
    single_game_frame = skimage.transform.resize(single_game_frame, (IMAGE_WIDTH, IMAGE_HEIGHT))
    single_game_frame = skimage.exposure.rescale_intensity(single_game_frame, out_range=(0, 255))

    for i in range(80):  # erasing the line in the right side of the screen
        single_game_frame[79, i] = 0

    # stacking up 4 images together to form a state: 4 images = state
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
    exploration_counter = 0
    exploration_flag = 0

    #j = 1
    os.mkdir("pic/", 0755);
    t = 0
    pic_counter = 0

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

        # choose an action epsilon greedy
        if (observation_counter % FRAME_PER_ACTION) == 0:
            if current_training_player == CurrentPlayer.Left:
                if random.random() <= epsilon:
                    action_index1 = random.randrange(NUM_OF_ACTIONS)
                    action_left_player[action_index1] = 1
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

            elif current_training_player == CurrentPlayer.Right:
                if random.random() <= epsilon or exploration_flag == 1:
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

        single_game_frame_colored, reward1, reward2, terminal, score, _ = \
            game_state.frame_step(action_left_player, action_right_player)

        single_game_frame_gray = skimage.color.rgb2gray(single_game_frame_colored)
        thresh = threshold_otsu(single_game_frame_gray)
        single_game_frame = single_game_frame_gray > thresh

        single_game_frame = skimage.transform.resize(single_game_frame, (80, 80))
        single_game_frame = skimage.exposure.rescale_intensity(single_game_frame, out_range=(0, 255))

        for i in range(80):  # erasing the line in the right side of the screen
            single_game_frame[79, i] = 0

        # if (score >= 0):
        #     fig1 = plt.figure(pic_counter)
        #     # plt.imshow(x_t1_colored)
        #     plt.imshow(single_game_frame)
        #     print('time now: ', datetime.datetime.now())
        # #fig1.savefig('pic/' + str(j) + '/' + str(pic_counter) + 'colored pic.png')
        # fig1.savefig('pic/' + str(pic_counter) + 'colored pic.png')
        #
        # plt.close()
        #
        # t = t + 1
        # pic_counter += 1

        single_game_frame = single_game_frame.reshape(1, 1, single_game_frame.shape[0],
                                                      single_game_frame.shape[1])

        # next 4 images = next state
        next_state = np.append(single_game_frame, current_state[:, :3, :, :], axis=1)

        if current_training_player == CurrentPlayer.Left:
            D1.append((current_state, action_index1, reward1, next_state, terminal))
            if len(D1) > REPLAY_MEMORY:
                D1.popleft()

        elif current_training_player == CurrentPlayer.Right:
            D2.append((current_state, action_index2, reward2, next_state, terminal))
            if len(D2) > REPLAY_MEMORY:
                D2.popleft()

        # only train if done observing
        if observation_counter > OBSERVATION:
            # sample a minibatch to train on - eliminates states correlation
            minibatch = None
            if current_training_player == CurrentPlayer.Left:
                minibatch = random.sample(D1, BATCH)
            elif current_training_player == CurrentPlayer.Right:
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

                if (current_training_player == CurrentPlayer.Left):
                    targets[i] = left_player.model.predict(current_state_t)
                    Q_sa = left_player.model.predict(next_state_t)
                elif (current_training_player == CurrentPlayer.Right):
                    targets[i] = right_player.model.predict(current_state_t)
                    Q_sa = right_player.model.predict(next_state_t)

                if terminal_t:
                    targets[i, action_t] = reward_t
                else:
                    targets[i, action_t] = reward_t + GAMMA * np.max(Q_sa)

            if (current_training_player == CurrentPlayer.Left):
                loss += left_player.model.train_on_batch(inputs, targets)
            elif (current_training_player == CurrentPlayer.Right):
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
            if current_training_player == CurrentPlayer.Left:
                left_player.model.save_weights('model1.h5', overwrite=True)
                with open('model1.json', 'w') as outfile:
                    json.dump(left_player.model.to_json(), outfile)

            elif current_training_player == CurrentPlayer.Right:
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

            print('game ended:\n')
            print('right player wins in a row: ', right_player.num_of_wins_in_a_row, '\n')
            print('left player wins in a row: ', left_player.num_of_wins_in_a_row, '\n')

            if (current_training_player == CurrentPlayer.Left and left_player.num_of_wins_in_a_row == 30):
                _ = save_weights_file(num_folder, current_training_player, left_player, right_player)
                # plot_loss(current_log_folder, current_log_folder + '/loss')
                # subprocess.call('. ~/flappy/bin/activate && ' + TEST_SEQUENTIAL_COMMAND + ' ' + str(1) +
                #                 ' ' + str(left_player.num_of_trains) + ' ' + str(right_player.num_of_trains),
                #                 shell=True)

                left_player.num_of_wins_in_a_row = 0
                right_player.num_of_wins_in_a_row = 0
                observation_counter = 0
                epsilon = INITIAL_EPSILON
                num_folder = 0
                current_training_player = CurrentPlayer.Right
                right_player.num_of_trains += 1
                current_log_folder = 'logs/' + 'right_player' + \
                                     '_learning' + str(right_player.num_of_trains)
                episode_number = 0
                D1.clear()
                start_time = datetime.datetime.now()
                break

            elif (current_training_player == CurrentPlayer.Right and right_player.num_of_wins_in_a_row == 30):
                # plot_loss(current_log_folder, current_log_folder + '/loss')
                _ = save_weights_file(num_folder, current_training_player, left_player, right_player)
                # subprocess.call('. ~/flappy/bin/activate && ' + TEST_SEQUENTIAL_COMMAND + ' ' + str(2) +
                #                 ' ' + str(left_player.num_of_trains) + ' ' + str(right_player.num_of_trains),
                #                 shell=True)

                left_player.num_of_wins_in_a_row = 0
                right_player.num_of_wins_in_a_row = 0
                observation_counter = 0
                epsilon = INITIAL_EPSILON
                num_folder = 0
                current_training_player = CurrentPlayer.Left
                left_player.num_of_trains += 1
                current_log_folder = 'logs/' + 'left_player' + \
                                     '_learning' + str(left_player.num_of_trains)
                episode_number = 0
                D2.clear()
                start_time = datetime.datetime.now()
                break

    print("Episode finished!")
    print("************************")


def train_simultaneously(left_player, right_player):
    if os.path.exists('logs_simultaneously'):
        if os.path.exists('old_logs_simultaneously'):
            shutil.rmtree('old_logs_simultaneously')
        os.mkdir('old_logs_simultaneously', 0755)
        copytree('logs_simultaneously', 'old_logs_simultaneously')
        shutil.rmtree('logs_simultaneously')
    os.mkdir('logs_simultaneously', 0755)

    # moving old trials to old_trials folder
    if os.path.exists('trials_simultaneously'):
        if os.path.exists('old_trials_simultaneously'):
            shutil.rmtree('old_trials_simultaneously')
        os.mkdir('old_trials_simultaneously', 0755)
        copytree('trials_simultaneously', 'old_trials_simultaneously')
        shutil.rmtree('trials_simultaneously')
    os.mkdir('trials_simultaneously', 0755)

    current_training_player = CurrentPlayer.Both
    left_player.num_of_trains += 1
    right_player.num_of_trains += 1

    # open up a game state to communicate with emulator
    game_state = game_sim_training.GameState()

    # store the previous observations in replay memory
    D1 = deque()
    D2 = deque()

    # get the first state (do nothing) and pre-process the image to 4x80x80
    do_nothing = np.zeros(NUM_OF_ACTIONS)
    do_nothing[0] = 1

    single_game_frame, _, _, terminal, _, _ = game_state.frame_step(do_nothing, do_nothing)

    single_game_frame = skimage.color.rgb2gray(single_game_frame)
    single_game_frame = skimage.transform.resize(single_game_frame, (IMAGE_WIDTH, IMAGE_HEIGHT))
    single_game_frame = skimage.exposure.rescale_intensity(single_game_frame, out_range=(0, 255))

    for i in range(80):  # erasing the line in the right side of the screen
        single_game_frame[79, i] = 0

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
        loss_left = 0
        loss_right = 0

        # player1 - left player
        action_index1 = 0
        action_left_player = np.zeros([NUM_OF_ACTIONS])

        # player2 - right player
        action_index2 = 0
        action_right_player = np.zeros([NUM_OF_ACTIONS])

        # choose an action epsilon greedy
        if (observation_counter % FRAME_PER_ACTION) == 0:
            if random.random() <= epsilon:
                action_index1 = random.randrange(NUM_OF_ACTIONS)
                action_left_player[action_index1] = 1
            else:
                q = left_player.model.predict(current_state)  # input a stack of 4 images, get the prediction
                max_Q = np.argmax(q)
                action_index1 = max_Q
                action_left_player[action_index1] = 1

            if random.random() <= epsilon:
                action_index2 = random.randrange(NUM_OF_ACTIONS)
                action_right_player[action_index2] = 1
            else:
                q = right_player.model.predict(current_state)  # input a stack of 4 images, get the prediction
                max_Q = np.argmax(q)
                action_index2 = max_Q
                action_right_player[action_index2] = 1

        # we reduced the epsilon gradually
        if (epsilon > FINAL_EPSILON) and (observation_counter > OBSERVATION):
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        single_game_frame_colored, left_player_reward, right_player_reward, terminal, score, _ = \
            game_state.frame_step(action_left_player, action_right_player)

        single_game_frame_gray = skimage.color.rgb2gray(single_game_frame_colored)
        thresh = threshold_otsu(single_game_frame_gray)
        single_game_frame = single_game_frame_gray > thresh

        single_game_frame = skimage.transform.resize(single_game_frame, (80, 80))
        single_game_frame = skimage.exposure.rescale_intensity(single_game_frame, out_range=(0, 255))
        single_game_frame = single_game_frame.reshape(1, 1, single_game_frame.shape[0],
                                                      single_game_frame.shape[1])

        for i in range(80):  # erasing the line in the right side of the screen
            single_game_frame[79, i] = 0

        # next 4 images = next state
        next_state = np.append(single_game_frame, current_state[:, :3, :, :], axis=1)

        D1.append((current_state, action_index1, left_player_reward, next_state, terminal))
        if len(D1) > REPLAY_MEMORY:
            D1.popleft()

        D2.append((current_state, action_index2, right_player_reward, next_state, terminal))
        if len(D2) > REPLAY_MEMORY:
            D2.popleft()

        # only train if done observing
        if observation_counter > OBSERVATION:
            # sample a minibatch to train on - eliminates states correlation
            minibatch1 = random.sample(D1, BATCH)
            inputs1 = np.zeros((BATCH, current_state.shape[1], current_state.shape[2],
                                current_state.shape[3]))  # 32, 4, 80, 80
            targets1 = np.zeros((inputs1.shape[0], NUM_OF_ACTIONS))  # 32, 2

            # experience replay
            for i in range(0, len(minibatch1)):
                current_state_t = minibatch1[i][0]
                action_t = minibatch1[i][1]  # This is action index
                reward_t = minibatch1[i][2]
                next_state_t = minibatch1[i][3]
                terminal_t = minibatch1[i][4]

                inputs1[i:i + 1] = current_state_t

                targets1[i] = left_player.model.predict(current_state_t)
                Q_sa = left_player.model.predict(next_state_t)

                if terminal_t:
                    targets1[i, action_t] = reward_t
                else:
                    targets1[i, action_t] = reward_t + GAMMA * np.max(Q_sa)

            loss_left += left_player.model.train_on_batch(inputs1, targets1)

            # sample a minibatch to train on - eliminates states correlation
            minibatch2 = random.sample(D2, BATCH)

            inputs2 = np.zeros((BATCH, current_state.shape[1], current_state.shape[2],
                                current_state.shape[3]))  # 32, 4, 80, 80
            targets2 = np.zeros((inputs2.shape[0], NUM_OF_ACTIONS))  # 32, 2

            # experience replay
            for i in range(0, len(minibatch2)):
                current_state_t = minibatch2[i][0]
                action_t = minibatch2[i][1]  # This is action index
                reward_t = minibatch2[i][2]
                next_state_t = minibatch2[i][3]
                terminal_t = minibatch2[i][4]

                inputs2[i:i + 1] = current_state_t

                targets2[i] = right_player.model.predict(current_state_t)
                Q_sa = right_player.model.predict(next_state_t)

                if terminal_t:
                    targets2[i, action_t] = reward_t
                else:
                    targets2[i, action_t] = reward_t + GAMMA * np.max(Q_sa)

            loss_right += right_player.model.train_on_batch(inputs2, targets2)

        current_state = next_state
        observation_counter = observation_counter + 1

        # save progress (updating weights files) every 100 iterations
        if observation_counter % 100 == 0:
            left_player.model.save_weights('model1.h5', overwrite=True)
            with open('model1.json', 'w') as outfile:
                json.dump(left_player.model.to_json(), outfile)

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
        first_learning_player = CurrentPlayer.Left
    elif int(args['first_learning_player']) == 2:
        first_learning_player = CurrentPlayer.Right

    if args['learning_mode'] == 'SEQUENTIALLY':  # one after another (left and then right and then left...)
        train_sequentially(left_player, right_player, first_learning_player)
    elif args['learning_mode'] == 'SIMULTANEOUSLY':
        train_simultaneously(left_player, right_player)


def main():
    parser = argparse.ArgumentParser(description='Ping-Pong Q-Learning')
    parser.add_argument('-l', '--learning_mode', help='SEQUENTIALLY, SIMULTANEOUSLY', required=True)
    parser.add_argument('-p', '--first_learning_player', help='1, 2', required=False)
    args = vars(parser.parse_args())
    play_game(args)


if __name__ == "__main__":
    main()
