import os
import shutil
from aenum import Enum


class CurrentPlayer(Enum):
    left = 1
    right = 2
    both = 3


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

    if current_training_player == CurrentPlayer.left or current_training_player == CurrentPlayer.both:
        os.makedirs('trials_sequentially/' + 'left_player_learning' +
                    str(left_player.num_of_trains) + '/' + str(num_folder), 0755)

        shutil.copy2('./model1.h5', 'trials_sequentially/' + 'left_player_learning' +
                     str(left_player.num_of_trains) + '/' + str(num_folder) + '/model1.h5')

    elif current_training_player == CurrentPlayer.right or current_training_player == CurrentPlayer.both:
        os.makedirs('trials_sequentially/' + 'right_player_learning' +
                    str(right_player.num_of_trains) + '/' + str(num_folder), 0755)

        shutil.copy2('./model2.h5', 'trials_sequentially/' + 'right_player_learning' +
                     str(right_player.num_of_trains) + '/' + str(num_folder) + '/model2.h5')

    return num_folder
