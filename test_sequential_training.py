import os
import sys
import run_test
from graph_utils import plot_scores, plot_times, plot_qmax, print_average_game_time


def test_player_after_switch(arg_learning_mode, arg_player1_num_of_trains, arg_player2_num_of_trains):
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
        test_player_folder_path = "./trials_sequentially/" + "left_player" + \
                                  "_learning" + str(player1_num_of_trains)
        print ('DEBUG: test_player_folder_path: ', test_player_folder_path, '\n')

        test_player_log_file = "logs/" + "left_player" + \
                               "_learning" + str(player1_num_of_trains)

    elif (learning_mode == 2):
        opponents_number = 1
        opponent = 'left'
        opponent_num_of_trains = player1_num_of_trains
        test_player_folder_path = "./trials_sequentially/" + "right_player" + \
                                  "_learning" + str(player2_num_of_trains)
        test_player_log_file = "logs/" + "right_player" + \
                               "_learning" + str(player2_num_of_trains)

    # opponent_path_to_most_updated_weights_folder = "./trials_sequentially/" + opponent + "_player" + \
    #                                                "_learning" + str(opponent_num_of_trains)
    # opponent_path_to_most_updated_weights_folder = "./trials_sequentially/" + "player" + str(opponents_number) +\
    #                                               "learning" + str(opponent_num_of_trains)
    # list_of_files = glob.glob(opponent_path_to_most_updated_weights_folder + '/*/*')
    # opponent_most_updated_weights_file = max(list_of_files, key=os.path.getctime)

    opponent_most_updated_weights_file = 'model' + str(opponents_number) + '.h5'

    num_of_test = 0
    game_times = []
    game_numbers = []
    times = []
    scores = []
    game_scores = []
    left_player_q_max_list = []
    left_player_q_max_list = []

    for subdir, dirs, files in os.walk(test_player_folder_path):
        dirs.sort(key=lambda f: int(filter(str.isdigit, f)))
        for d in dirs:
            num_of_test += 1
            if learning_mode == 1:
                times, scores, left_player_q_max_list, right_player_q_max_list = \
                    run_test.main(1, test_player_folder_path + "/" + str(d) + "/model1.h5",
                                  opponent_most_updated_weights_file,
                                  num_of_test, test_player_log_file)
                game_times = game_times + times
                game_scores = game_scores + scores
            elif learning_mode == 2:
                times, scores, left_player_q_max_list, right_player_q_max_list = \
                    run_test.main(2, opponent_most_updated_weights_file,
                                  test_player_folder_path + "/" + str(d) + "/model2.h5",
                                  num_of_test, test_player_log_file)
                game_times = game_times + times
                game_scores = game_scores + scores

    game_numbers.extend(range(1, len(game_times) + 1))
    plot_qmax(test_player_log_file, test_player_log_file + '/qmax')
    plot_scores(game_numbers, game_scores, test_player_log_file)
    plot_times(game_numbers, game_times, test_player_log_file)
    print_average_game_time(test_player_log_file)


def main(arg_learning_mode, arg_player1_num_of_trains, arg_player2_num_of_trains):
    test_player_after_switch(arg_learning_mode, arg_player1_num_of_trains, arg_player2_num_of_trains)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2], sys.argv[3]))
