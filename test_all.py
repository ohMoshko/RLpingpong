import os
import sys
import glob
import run_test
import argparse

def testPlayerAfterSwitch(arg_learning_mode, arg_player1_num_of_trains, arg_player2_num_of_trains):
    opponents_number = 0

    learning_mode = int(arg_learning_mode)
    player1_num_of_trains = int(arg_player1_num_of_trains)
    player2_num_of_trains = int(arg_player2_num_of_trains)


    opponent_num_of_trains = 0
    if (learning_mode == 1):
        opponents_number = 2
        opponent_num_of_trains = player2_num_of_trains
    elif (learning_mode == 2):
        opponents_number = 1
        opponent_num_of_trains = player1_num_of_trains


    opponent_path_to_most_updated_weights_folder = "./trials/" + "player" + str(opponents_number) +\
                                          "learning" + str(opponent_num_of_trains)


    list_of_files = glob.glob(opponent_path_to_most_updated_weights_folder + '/*/*')

    opponent_most_updated_weights_file = max(list_of_files, key=os.path.getctime)

    test_player_folder_path = "./trials/" + "player" + str(learning_mode) +\
                         "learning" + str(opponent_num_of_trains)

    test_player_log_file = "logs/" + "player" + str(learning_mode) +\
                         "learning" + str(opponent_num_of_trains)

    num_of_test = 0

    for subdir, dirs, files in os.walk(test_player_folder_path):
        dirs.sort(key=lambda f: int(filter(str.isdigit, f)))
        for d in dirs:
            for subdir2, dirs2, files2 in os.walk(test_player_folder_path + "/" + str(d)):
                num_of_test = num_of_test + 1
                if (learning_mode == 1):
                    run_test.main(file, opponent_most_updated_weights_file, num_of_test, test_player_log_file)
                elif (learning_mode == 2):
                    run_test.main(opponent_most_updated_weights_file, file, num_of_test, test_player_log_file)


def main(arg_learning_mode, arg_player1_num_of_trains, arg_player2_num_of_trains):
    testPlayerAfterSwitch(arg_learning_mode, arg_player1_num_of_trains, arg_player2_num_of_trains)

if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2], sys.argv[3]))
