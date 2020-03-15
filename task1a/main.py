#    _____         .__
#   /     \ _____  |__| ____
#  /  \ /  \\__  \ |  |/    \
# /    Y    \/ __ \|  |   |  \
# \____|__  (____  /__|___|  /
#         \/     \/        \/
'''The main file is used to start up the program and initialize the needed objects'''

__author__ = "Manuel Galliker"
__maintainer__ = "Manuel Galliker"
__license__ = "GPL"

from src import FileManager
from src import RowVector
from src import minimize_ridge_regression as mrr
import pandas as pd
import statistics


def main():

    file_manager = FileManager()

    ### Training Data Initialization ###
    training_data_pd = file_manager.load_csv('data/train.csv')
    training_data = training_data_pd.to_numpy()
    # matrix containing all training feature data
    training_features = training_data[:, 2:]
    # vector containing all output labels
    training_output = training_data[:,1]

    reg_param_list = [0.01, 0.1, 1, 10, 100]

    for reg_param in reg_param_list:
        optimal_weights = mrr.minimize_ridge_regression(training_output, training_features, reg_param)
        print(optimal_weights)
    return


if __name__ == "__main__":

    main()
