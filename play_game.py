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
import time
import pygame

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


def play_game(left_player):

    # open up a game state to communicate with emulator
    game_state = game.GameState()
    time.sleep(2)
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
    last_4_frames = last_4_frames.reshape(1, last_4_frames.shape[0],
                                          last_4_frames.shape[1], last_4_frames.shape[2])

    number_of_games = 3

    while number_of_games > 0:
        actions_vector1 = np.zeros([ACTIONS])
        actions_vector2 = np.zeros([ACTIONS])

        # choose an action for the left player:
        q1 = left_player.model.predict(last_4_frames)  # input a stack of 4 images, get the prediction
        max_Q1 = np.argmax(q1)
        action_index1 = max_Q1
        actions_vector1[action_index1] = 1

        # action for right player: input from a human - keyboard
        # events = pygame.event.get()
        # for event in events:
        #     if event.type == pygame.KEYDOWN:
        #         if event.key == pygame.K_UP:
        #             #action_index2 = 1
        #             actions_vector2[1] = 1
        #         if event.key == pygame.K_DOWN:
        #             #action_index2 = 2
        #             actions_vector2[2] = 1
        #         if event.key == pygame.K_ESCAPE:
        #             exit()

        keys = pygame.key.get_pressed()  # checking pressed keys
        if keys[pygame.K_UP]:
            actions_vector2[1] = 1
        if keys[pygame.K_DOWN]:
            actions_vector2[2] = 1

        #actions_vector2[action_index2] = 1

        # in order for us to see the game
        image_data_colored1, _, _, terminal, score, no_learning_time =\
            game_state.frame_step(actions_vector1, actions_vector2)

        game_over = terminal
        if game_over:
            time.sleep(5)

        # image processing
        x_t1 = skimage.color.rgb2gray(image_data_colored1)
        x_t1 = skimage.transform.resize(x_t1, (80, 80))
        x_t1 = skimage.exposure.rescale_intensity(x_t1, out_range=(0, 255))

        for i in range(80):  # erasing the line in the right side of the screen
            x_t1[79, i] = 0

        x_t1 = x_t1.reshape(1, 1, x_t1.shape[0], x_t1.shape[1])
        last_4_frames1 = np.append(x_t1, last_4_frames[:, :3, :, :], axis=1)

        last_4_frames = last_4_frames1

    print('game over!\n')


def main(weights1_file):
    left_player = Player()
    left_player.build_model()
    left_player.load_model_weights(weights1_file)
    play_game(left_player)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
