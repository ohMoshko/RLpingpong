#!/usr/bin/env python
# !/usr/bin/env python
from __future__ import print_function

import argparse
from skimage.transform import rotate
from skimage.viewer import ImageViewer
import sys
import numpy as np
import matplotlib.pyplot as plt
import linecache
import numpy as np

import matplotlib.pyplot as plt
from operator import itemgetter

GAME = "pong"


def plot_loss(loss_folder, loss_file):
    # loss

    logfile8 = open(loss_file, 'r')

    loss_content = logfile8.read()
    points8 = [(int(line.split()[0]), float(line.split()[2])) for line in loss_content.splitlines()]

    plt.plot([p[0] for p in points8], [p[1] for p in points8], lw=2, zorder=2, label="loss")

    fig4 = plt.figure(1)
    plt.xlabel('Episode')
    plt.ylabel('loss')

    plt.title('Training: agent loss')
    #plt.show()

    fig4.savefig(loss_folder + '/loss_plot.png')
    plt.clf()


def plot_qmax(qmax_folder, qmax_file):
    # qmax

    num_of_different_tests = 0
    with open(qmax_file, 'r') as f:
        last_line_in_qmax_log_file = f.readlines()[-1]
        num_of_different_tests = int(last_line_in_qmax_log_file.split()[0])

    qmax_log_file = open(qmax_file, 'r')
    q_max_content = qmax_log_file.read()
    points7 = [(int(line.split()[0]), float(line.split()[2])) for line in q_max_content.splitlines()]
    print(len(points7, ))
    points7 = points7[::5]
    print(len(points7, ))
    plt.plot([p[0] for p in points7], [p[1] for p in points7], '+', zorder=1, label="per step")

    average_q_max = []
    i = 1
    sum = 0
    total_samples = 0
    for line in q_max_content.splitlines():
        if int(line.split()[0]) == i:
            sum += float(line.split()[2])
            total_samples += 1
        else:
            average_q_max.append((i, sum/total_samples))
            total_samples = 1
            i += 1
            sum = float(line.split()[2])
    average_q_max.append((i, sum / total_samples))

    plt.plot([p[0] for p in average_q_max], [p[1] for p in average_q_max], lw=2, zorder=2, label="average qmax")
    plt.xlim(0, num_of_different_tests + 1)
    plt.legend(loc=4)

    fig5 = plt.figure(1)
    plt.xlabel('Match')
    plt.ylabel('qmax')

    plt.title('Hit Reward = 0.2+negative: Agent q max')
    #plt.show()

    fig5.savefig(qmax_folder + '/qmax_plot.png')
    plt.clf()


def plot_scores(game_numbers, game_scores, test_player_log_file):
    plt.plot(game_numbers, [gs[0] for gs in game_scores], 'bo', zorder=1, label="left player")
    plt.plot(game_numbers, [gs[0] for gs in game_scores], 'k')
    plt.plot(game_numbers, [gs[1] for gs in game_scores], 'ro', zorder=2, label="right player")
    plt.plot(game_numbers, [gs[1] for gs in game_scores], 'k')
    plt.legend(loc=1)
    plt.title('Hit Reward = 0.2+negative: Game Score')
    plt.xlabel('game number')
    plt.ylabel('game scores')
    plt.yticks(range(1, 25))
    plt.savefig(test_player_log_file + '/game_scores_vs_game_num.png')
    plt.clf()


def plot_times(game_numbers, game_times, test_player_log_file):
    game_numbers.extend(range(1, len(game_times) + 1))
    plt.plot(game_numbers, game_times, 'b')
    # plt.plot(game_numbers, [gs[0] + gs[1] for gs in game_scores], 'r', zorder=2, label='score')
    plt.ylim(0)
    # plt.legend(loc=4)
    plt.title('Hit Reward = 0.2+negative: Game Time')
    plt.xlabel('game number')
    plt.ylabel('game time [sec]')
    plt.savefig(test_player_log_file + '/game_time_vs_game_num.png')
    plt.clf()


def print_correlation():
    # qmax
    logfile6 = open('logs_pong/Corelation_ratio1.txt', 'r')

    correlation = logfile6.read()
    points6 = [(int(line.split()[0]), float(line.split()[3])) for line in correlation.splitlines()]

    plt.plot([p[0] for p in points6], [p[1] for p in points6], lw=2, zorder=2, label="average qmax")

    fig5 = plt.figure(1)
    plt.xlabel('Epoch')
    plt.ylabel('Correlation ratio')

    plt.title('Pong - correlation ratio - Hybrid q-learn ')
    plt.show()

    fig5.savefig(GAME + '_Hybrid_q_learn_correlation.png')
    plt.clf()

def print_scores():
    # scores
    logfile1 = open('logs_pong/resultsfile_1.txt', 'r')

    results = logfile1.read()
    points1 = [(int(line.split()[0]), float(line.split()[7])) for line in results.splitlines()]
    plt.plot([p[0] for p in points1], [p[1] for p in points1], 'o', zorder=1, label="per game")

    logfile2 = open('logs_pong/averagefile_1.txt', 'r')
    average = logfile2.read()
    points2 = [(int(line.split()[0]), float(line.split()[3])) for line in average.splitlines()]
    plt.plot([p[0] for p in points2], [p[1] for p in points2], lw=2, zorder=2, label="average")
    plt.legend(loc=4)

    fig1 = plt.figure(1)
    plt.xlabel('Epoch')
    plt.ylabel('win percentage')

    plt.title('Pong - agent scores - Hybrid q-learn ')
    plt.show()

    fig1.savefig(GAME + '_Hybrid_q_learn_results.png')
    plt.clf()


def print_rewards():
    # result
    logfile3 = open('logs_pong/rewardfile_1.txt', 'r')
    reward = logfile3.read()
    points3 = [(int(line.split()[0]), float(line.split()[3])) for line in reward.splitlines()]
    plt.plot([p[0] for p in points3], [p[1] for p in points3], 'o', zorder=1, label="per game")

    logfile4 = open('logs_pong/reward_finalfile_1.txt', 'r')
    average_reward = logfile4.read()
    points4 = [(int(line.split()[0]), float(line.split()[3])) for line in average_reward.splitlines()]

    plt.plot([p[0] for p in points4], [p[1] for p in points4], lw=2, zorder=2, label="average")
    fig2 = plt.figure(1)
    plt.xlabel('Epoch')
    plt.ylabel('reward')
    plt.legend(loc=4)

    plt.title('pong - agent reward - Hybrid q-learn ')
    plt.show()

    fig2.savefig(GAME + '_Hybrid_q_learn_reward.png')
    plt.clf()


def main():
    print_correlation()
    print_scores()
    print_rewards()
    print_loss()
    print_qmax()


if __name__ == "__main__":
    main()