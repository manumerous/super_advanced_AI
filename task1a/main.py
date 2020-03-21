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
from src import DataContainer
from src import GradientDescent
from src import ridge_regression_solver as rgs
from src import generate_cv_datasets as gcv
from src import calculate_rmse as rmse
import numpy as np
import pandas as pd
import statistics


def main():

    file_manager = FileManager()

    ### Training Data Initialization ###
    training_data_pd = file_manager.load_csv('data/train.csv')
    raw_data = training_data_pd.to_numpy()

    reg_param_list = [0.01, 0.1, 1, 10, 100]
    cross_validation_count = 10

    for reg_param in reg_param_list:

        weight_collector = np.zeros(13)
        rmse_collector = 0
        for i in range(cross_validation_count):

            raw_test_set, raw_train_set = gcv.generate_cv_datasets(
                cross_validation_count, i, raw_data)
            test_set = DataContainer(raw_test_set)
            train_set = DataContainer(raw_train_set)
            optimal_weights = rgs.minimize_ridge_regression(
                train_set.get_y(), train_set.get_x(), reg_param)
            weight_collector += optimal_weights
            # print(test_set.x.shape)
            y_hat = optimal_weights @ test_set.get_x().transpose()
            # print(y_hat)
            rmse_collector += rmse.calculate_rmse(test_set.get_y(), y_hat)

        averaged_weights = weight_collector/cross_validation_count
        averaged_rmse = rmse_collector/cross_validation_count
        # print(averaged_weights)
        print(averaged_rmse)

    return


if __name__ == "__main__":

    main()
