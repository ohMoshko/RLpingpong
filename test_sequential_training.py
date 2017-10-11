import os
import sys
import glob
import run_test
import argparse
import matplotlib.pyplot as plt

def testPlayerAfterSwitch(arg_learning_mode, arg_player1_num_of_trains, arg_player2_num_of_trains):

    opponents_number = 0
    learning_mode = int(arg_learning_mode)
    player1_num_of_trains = int(arg_player1_num_of_trains)
    player2_num_of_trains = int(arg_player2_num_of_trains)
    test_player_folder_path = ''
    test_player_log_file = ''
    opponent = ''
    opponent_num_of_trains = 0

    if (learning_mode == 1):
        opponents_number = 2
        opponent = 'right'
        opponent_num_of_trains = player2_num_of_trains
        test_player_folder_path = "./trials/" + "left_player" + \
                                  "_learning" + str(player1_num_of_trains)
        test_player_log_file = "logs/" + "right_player" + \
                               "_learning" + str(player1_num_of_trains)

    elif (learning_mode == 2):
        opponents_number = 1
        opponent = 'left'
        opponent_num_of_trains = player1_num_of_trains
        test_player_folder_path = "./trials/" + "player" + str(learning_mode) + \
                                  "learning" + str(player2_num_of_trains)
        test_player_log_file = "logs/" + "player" + str(learning_mode) + \
                               "learning" + str(player2_num_of_trains)

    opponent_path_to_most_updated_weights_folder = "./trials/" + opponent + "_player" +\
                                          "_learning" + str(opponent_num_of_trains)


    list_of_files = glob.glob(opponent_path_to_most_updated_weights_folder + '/*/*')

    opponent_most_updated_weights_file = max(list_of_files, key=os.path.getctime)

    num_of_test = 0
    game_times = []
    game_numbers = []
    times = []
    scores = []
    game_scores = []

    for subdir, dirs, files in os.walk(test_player_folder_path):
        dirs.sort(key=lambda f: int(filter(str.isdigit, f)))
        for d in dirs:
            for subdir2, dirs2, files2 in os.walk(test_player_folder_path + "/" + str(d)):
                num_of_test = num_of_test + 1
                if (learning_mode == 1):
                    times, scores = run_test.main(test_player_folder_path + "/" + d + "/" + files2[0],
                                          opponent_most_updated_weights_file,
                                          num_of_test, test_player_log_file)
                    game_times = game_times + times
                    game_scores = game_scores + scores
                elif (learning_mode == 2):
                    times, scores = run_test.main(opponent_most_updated_weights_file,
                                          test_player_folder_path + "/" + d + "/" + files2[0],
                                          num_of_test, test_player_log_file)
                    game_times = game_times + times
                    game_scores = game_scores + scores

    str_last_learning_player = ''
    last_learning_player_num_of_trains = 0
    if learning_mode == 1:
        str_last_learning_player = 'Left'
        last_learning_player_num_of_trains = player1_num_of_trains
    else:
        learning_mode = 'Right'
        last_learning_player_num_of_trains = player1_num_of_trains

    game_numbers.extend(range(1, len(game_times) + 1))
    plt.plot(game_numbers, game_times)
    plt.title(str_last_learning_player + ' player sequentially training number ' +
              str(last_learning_player_num_of_trains))
    plt.xlabel('gmae number')
    plt.ylabel('game length [sec]')
    plt.savefig(test_player_log_file + '/game_length_vs_game_num.png')

    plt.plot(game_numbers, [gs[0] for gs in game_scores], 'bo', zorder=1, label="left player")
    plt.plot(game_numbers, [gs[0] for gs in game_scores], 'k')
    plt.plot(game_numbers, [gs[1] for gs in game_scores], 'ro', zorder=2, label="right player")
    plt.plot(game_numbers, [gs[1] for gs in game_scores], 'k')
    plt.legend(loc=4)

    plt.title(str_last_learning_player + ' player sequentially training number ' +
              str(last_learning_player_num_of_trains) + 'game_scores')
    plt.xlabel('gmae number')
    plt.ylabel('game scores')
    plt.savefig(test_player_log_file + '/game_scores_vs_game_num.png')


def main(arg_learning_mode, arg_player1_num_of_trains, arg_player2_num_of_trains):
    testPlayerAfterSwitch(arg_learning_mode, arg_player1_num_of_trains, arg_player2_num_of_trains)

if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2], sys.argv[3]))