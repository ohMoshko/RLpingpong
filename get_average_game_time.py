#!/usr/bin/env python

from __future__ import print_function
import sys


def get_average_game_time(log_file):
    game_over_log_file = open(log_file, 'r')
    file_content = game_over_log_file.read()
    times = [round(float(line.split()[5]), 2) for line in file_content.splitlines()]
    avg = sum(times) / len(times)
    print('average game time: ', avg)


def main(log_file):
    get_average_game_time(log_file)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
