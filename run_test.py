#!/usr/bin/env python
from __future__ import print_function

import sys
import skimage as skimage
from skimage import transform, color, exposure
sys.path.append("game/")
import pong_fun as game
import numpy as np
import os.path
import datetime
from player import Player

GAME = 'pong'  # the name of the game being played for log files
CONFIG = 'nothreshold'
ACTIONS = 3  # number of valid actions
GAMMA = 0.99  # decay rate of past observations
OBSERVATION = 320.  # timesteps to observe before training
EXPLORE = 3000000.  # frames over which to anneal epsilon
REPLAY_MEMORY = 50000  # number of previous transitions to remember
BATCH = 32  # size of minibatch

img_rows, img_cols = 80, 80
# Convert image into Black and white
img_channels = 4  # We stack 4 frames


def run_test(learnig_player, left_player, right_player, num_of_test, test_player_log_file):
    if not os.path.exists(test_player_log_file):
        os.makedirs(test_player_log_file, 0755)

    test_start_time = datetime.datetime.now()
    no_learning_time = 0
    # open up a game state to communicate with emulator
    game_state = game.GameState()

    # get the first state by doing nothing and preprocess the image to 80x80x4
    do_nothing = np.zeros(ACTIONS)

    # Prevents error in frame step. Both bars stay in place.
    do_nothing[0] = 1
    game_image_data, _, r_0, terminal, _, _ = game_state.frame_step(do_nothing, do_nothing)

    # image processing
    game_image_data = skimage.color.rgb2gray(game_image_data)
    game_image_data = skimage.transform.resize(game_image_data, (80, 80))
    game_image_data = skimage.exposure.rescale_intensity(game_image_data, out_range=(0, 255))

    for i in range(80):  # erasing the line in the right side of the screen
        game_image_data[79, i] = 0

    # initiating first 4 frames to the same frame
    last_4_frames = np.stack((game_image_data, game_image_data, game_image_data, game_image_data), axis=0)

    # In Keras, need to reshape
    last_4_frames = last_4_frames.reshape(1, last_4_frames.shape[0], last_4_frames.shape[1], last_4_frames.shape[2])

    num_folder = 0
    left_player_scores = []
    right_player_scores = []
    time_list = []
    game_score = []
    left_player_q_max_list = []
    right_player_q_max_list = []

    number_of_games = 10
    #number_of_games = 10 #for epsilons tests

    original_number_of_games = number_of_games

    game_start_time = datetime.datetime.now()

    left_player.num_of_wins = 0
    right_player.num_of_wins = 0

    while (number_of_games > 0):
        actions_vector1 = np.zeros([ACTIONS])
        actions_vector2 = np.zeros([ACTIONS])

        # choose an action epsilon greedy for player 1:
        q1 = left_player.model.predict(last_4_frames)  # input a stack of 4 images, get the prediction
        max_Q1 = np.argmax(q1)
        action_index1 = max_Q1
        q_max1 = np.amax(q1)
        left_player_q_max_list.append((num_folder, q_max1))

        # actions_vector1: input vector for frame_step function. contains the desired action for player 1
        # e.g [0,1,0]- up
        actions_vector1[action_index1] = 1

        # choose an action epsilon greedy for player 2:
        q2 = right_player.model.predict(last_4_frames)  # input a stack of 4 images, get the prediction
        max_Q2 = np.argmax(q2)
        action_index2 = max_Q2
        q_max2 = np.amax(q2)
        right_player_q_max_list.append((num_folder, q_max2))

        # actions_vector2: input vector for frame_step function. contains the desired action
        # e.g [0,1,0]- up
        actions_vector2[action_index2] = 1
        with open(test_player_log_file + '/qmax', 'a') as qmax_log_file:
            if learnig_player == 1:
                qmax_log_file.write(str(num_of_test) + ' : ' + str(q_max1) + '\n')
            else:
                qmax_log_file.write(str(num_of_test) + ' : ' + str(q_max2) + '\n')

        # in order for us to see the game
        image_data_colored1, _, _, terminal, score, no_learning_time = game_state.frame_step(actions_vector1,
                                                                                          actions_vector2)

        game_over = terminal
        if (game_over == True):
            number_of_games -= 1
            print(str(datetime.datetime.now()) + " game ended:   " + str(number_of_games) + " games left for the test")

            if (score[0] > score[1]):
                left_player.num_of_wins += 1
            else:
                right_player.num_of_wins += 1

            with open(test_player_log_file + "/" + "game_over_log", "a") as game_over_file:
                game_over_file.write("score: " + str(score) +
                                     "   game duration: " +
                                     str((datetime.datetime.now() - game_start_time).total_seconds()
                                         - no_learning_time) +
                                     " [sec]" + "\n")
                game_over_file.flush()

            left_player_scores.append(score[0])
            right_player_scores.append(score[1])
            game_score.append(score)

            current_time = datetime.datetime.now()
            elapsedTime = (current_time - game_start_time).total_seconds() - no_learning_time
            time_list.append(elapsedTime)

            game_start_time = datetime.datetime.now()

        # image processing
        x_t1 = skimage.color.rgb2gray(image_data_colored1)
        x_t1 = skimage.transform.resize(x_t1, (80, 80))
        x_t1 = skimage.exposure.rescale_intensity(x_t1, out_range=(0, 255))

        for i in range(80):  # erasing the line in the right side of the screen
            x_t1[79, i] = 0

        x_t1 = x_t1.reshape(1, 1, x_t1.shape[0], x_t1.shape[1])
        last_4_frames1 = np.append(x_t1, last_4_frames[:, :3, :, :], axis=1)

        last_4_frames = last_4_frames1

    if (number_of_games == 0):
        average_time = np.mean(time_list)
        left_player_average_score = np.mean(left_player_scores)
        right_player_average_score = np.mean(right_player_scores)

        print("left_player_num_of_wins: ", left_player.num_of_wins)
        print("\nright_player_num_of_wins: ", right_player.num_of_wins)

        left_player_win_percentage = (left_player.num_of_wins / float(original_number_of_games)) * 100
        right_player_win_percentage = (right_player.num_of_wins / float(original_number_of_games)) * 100

        print("\n\nleft_player_win_percentage: ", left_player_win_percentage)
        print("\nright_player_win_percentage: ", right_player_win_percentage)

        with open(test_player_log_file + "/" + "game_summary", "a") as results_file:
            results_file.write("\n" +
                               "Game Summary" + str(num_of_test) + ":" + "\n" +
                               "   left player average score: " + str(left_player_average_score) + " [points]" + "\n" +
                               "   left player win percentage: " + str(left_player_win_percentage) + "%" + "\n" +
                               "   right player average score: " + str(
                right_player_average_score) + " [points]" + "\n" +
                               "   right player win percentage: " + str(right_player_win_percentage) + "%" + "\n" +
                               "   average time: " + str(average_time) + "[sec]" + "\n" + "\n" + "\n")

    return time_list, game_score, left_player_q_max_list, right_player_q_max_list


def main(learning_player, weights1_file, weights2_file, num_of_test, test_player_log_file):
    print(weights1_file, " ", weights2_file)

    left_player = Player()
    left_player.build_model()
    left_player.load_model_weights(weights1_file)
    right_player = Player()
    right_player.build_model()
    right_player.load_model_weights(weights2_file)

    return run_test(learning_player, left_player, right_player, num_of_test, test_player_log_file)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]))
