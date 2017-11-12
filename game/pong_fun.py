#!/usr/bin/env python
# Modified from http://www.pygame.org/project-Very+simple+Pong+game-816-.html

import datetime
import os
import random
import pygame
from pygame.locals import *

position = 5, 325
os.environ['SDL_VIDEO_WINDOW_POS'] = str(position[0]) + "," + str(position[1])
pygame.init()
screen = pygame.display.set_mode((640, 480), 0, 32)
# screen = pygame.display.set_mode((640,480),pygame.NOFRAME)
# Creating 2 bars, a ball and background.
# background
back = pygame.Surface((640, 480))
background = back.convert()
background.fill((0, 0, 0))
# bars
bar = pygame.Surface((10, 50))
bar1 = bar.convert()
bar1.fill((0, 255, 255))
bar2 = bar.convert()
bar2.fill((255, 255, 255))
# ball
circ_sur = pygame.Surface((15, 15))
circ = pygame.draw.circle(circ_sur, (255, 255, 255), (15 / 2, 15 / 2), 15 / 2)
circle = circ_sur.convert()
circle.set_colorkey((0, 0, 0))
font = pygame.font.SysFont("calibri", 40)

ai_speed = 15.
ai_speed2 = 7.
HIT_REWARD = 0.3
LOSE_REWARD = -1
SCORE_REWARD = 1
#HIT_REWARD_AFTER_MAXIMUM_HITS = -0.1
#INITIAL_HIT_COUNTER_VALUE = 0
#MAXIMUM_HITS_PER_POINT = 5


class GameState:
    def __init__(self):
        self.bar1_x, self.bar2_x = 10., 620.
        self.bar1_y, self.bar2_y = 215., 215.
        self.circle_x, self.circle_y = 307.5, 232.5
        self.bar1_move, self.bar2_move = 0., 0.
        self.bar1_score, self.bar2_score = 0, 0
        self.speed_x, self.speed_y = 7., 7.
        pygame.key.set_repeat(1, 10)
        self.current_time = datetime.datetime.now()
        self.last_score_time = datetime.datetime.now()
        self.no_learning_time = 0
        self.left_player_hits_counter = 0
        self.right_player_hits_counter = 0

    def init_after_game_is_stuck(self):
        self.last_score_time = self.current_time
        if (self.speed_y < 1 and self.speed_y > -1):
            self.speed_y = -self.speed_y * (random.uniform(-17, 17))
        elif (self.speed_y > 3 or self.speed_y < -3):
            self.speed_y = -self.speed_y * (random.uniform(-1, 1))
        else:
            self.speed_y = -self.speed_y * (random.uniform(-2.5, 2.5))

    def frame_step(self, input_vect, input_vect2):
        global score
        pygame.event.pump()
        left_player_reward = 0
        right_player_reward = 0

        if sum(input_vect) != 1:
            raise ValueError('Multiple input actions!')

        if input_vect[1] == 1:  # Key up
            self.bar1_move = -ai_speed
        elif input_vect[2] == 1:  # Key down
            self.bar1_move = ai_speed
        else:  # don't move
            self.bar1_move = 0

        self.score1 = font.render(str(self.bar1_score), True, (255, 255, 255))
        self.score2 = font.render(str(self.bar2_score), True, (255, 255, 255))

        screen.blit(background, (0, 0))
        frame = pygame.draw.rect(screen, (255, 255, 255), Rect((5, 5), (630, 470)), 2)
        middle_line = pygame.draw.aaline(screen, (255, 255, 255), (330, 5), (330, 475))
        screen.blit(bar1, (self.bar1_x, self.bar1_y))
        screen.blit(bar2, (self.bar2_x, self.bar2_y))
        screen.blit(circle, (self.circle_x, self.circle_y))
        #screen.blit(self.score1, (250., 210.))
        #screen.blit(self.score2, (380., 210.))

        self.bar1_y += self.bar1_move

        if input_vect2[1] == 1:  # Key up
            self.bar2_y -= ai_speed2
        elif input_vect2[2] == 1:  # Key down
            self.bar2_y += ai_speed2
        else:  # don't move
            self.bar2_y += 0

        # bounds of movement
        if self.bar1_y >= 420.:
            self.bar1_y = 420.
        elif self.bar1_y <= 10.:
            self.bar1_y = 10.
        if self.bar2_y >= 420.:
            self.bar2_y = 420.
        elif self.bar2_y <= 10.:
            self.bar2_y = 10.

        # since i don't know anything about collision, ball hitting bars goes like this.
        if self.circle_x <= self.bar1_x + 10.:
            if self.circle_y >= self.bar1_y - 7.5 and self.circle_y <= self.bar1_y + 42.5:
                self.current_time = datetime.datetime.now()
                if ((self.current_time - self.last_score_time).total_seconds() > 90):
                    self.init_after_game_is_stuck()
                    self.no_learning_time = self.no_learning_time + 90
                # self.speed_y = 30.
                self.circle_x = 20.
                self.speed_x = -self.speed_x
                #if self.left_player_hits_counter >= MAXIMUM_HITS_PER_POINT:
                #    left_player_reward = HIT_REWARD_AFTER_MAXIMUM_HITS
                #else:
                #    self.left_player_hits_counter += 1
                left_player_reward = HIT_REWARD

        if self.circle_x >= self.bar2_x - 15.:
            if self.circle_y >= self.bar2_y - 7.5 and self.circle_y <= self.bar2_y + 42.5:
                self.current_time = datetime.datetime.now()
                if ((self.current_time - self.last_score_time).total_seconds() > 90):
                    self.init_after_game_is_stuck()
                    self.no_learning_time = self.no_learning_time + 90
                # self.speed_y = 70.
                self.circle_x = 605.
                self.speed_x = -self.speed_x
                #if self.right_player_hits_counter >= MAXIMUM_HITS_PER_POINT:
                 #   right_player_reward = HIT_REWARD_AFTER_MAXIMUM_HITS
                #else:
                #    self.right_player_hits_counter += 1
                right_player_reward = HIT_REWARD

        # scoring
        if self.circle_x < 5.:
            self.last_score_time = datetime.datetime.now()
            self.bar2_score += 1
            left_player_reward = LOSE_REWARD
            right_player_reward = SCORE_REWARD
            self.left_player_hits_counter = 0
            self.right_player_hits_counter = 0
            self.circle_x, self.circle_y = 307.5, 232.5
            self.speed_x = -self.speed_x
            # added randomality after a hit for learning purposes
            if (self.speed_y < 1 and self.speed_y > -1):
                self.speed_y = -self.speed_y * (random.uniform(-17, 17))
            elif (self.speed_y > 3 or self.speed_y < -3):
                self.speed_y = -self.speed_y * (random.uniform(-1, 1))
            else:
                self.speed_y = -self.speed_y * (random.uniform(-2.5, 2.5))

            self.bar1_y, self.bar_2_y = 215., 215.

        elif self.circle_x > 620.:
            self.last_score_time = datetime.datetime.now()
            self.bar1_score += 1
            right_player_reward = LOSE_REWARD
            left_player_reward = SCORE_REWARD
            self.left_player_hits_counter = 0
            self.right_player_hits_counter = 0
            self.circle_x, self.circle_y = 320., 232.5
            self.speed_x = -self.speed_x
            # added randomality after a hit for learning purposes
            if (self.speed_y < 1 and self.speed_y > -1):
                self.speed_y = -self.speed_y * (random.uniform(-17, 17))
            elif (self.speed_y > 3 or self.speed_y < -3):
                self.speed_y = -self.speed_y * (random.uniform(-1, 1))
            else:
                self.speed_y = -self.speed_y * (random.uniform(-2.5, 2.5))

            self.bar1_y, self.bar2_y = 215., 215.

        # collisions on sides
        if self.circle_y <= 10.:
            self.speed_y = -self.speed_y
            self.circle_y = 10.
        elif self.circle_y >= 457.5:
            self.speed_y = -self.speed_y
            self.circle_y = 457.5

        self.circle_x += self.speed_x
        self.circle_y += self.speed_y

        #screen.blit(background, (250., 210.))
        #screen.blit(background, (380., 210.))
        image_data = pygame.surfarray.array3d(pygame.display.get_surface())
        screen.blit(self.score1, (250., 210.))
        screen.blit(self.score2, (380., 210.))
        pygame.display.update()

        terminal = False

        no_learning_time_to_return = self.no_learning_time

        if max(self.bar1_score, self.bar2_score) >= 20:
            self.no_learning_time = 0
            score = self.bar1_score, self.bar2_score
            self.bar1_score = 0
            self.bar2_score = 0
            terminal = True

        else:
            score = self.bar1_score, self.bar2_score

        return image_data, left_player_reward, right_player_reward, terminal, score, no_learning_time_to_return
