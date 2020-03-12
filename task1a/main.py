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
from src import rsme_function as rsme_function
import pandas as pd
import statistics


def main():

    file_manager = FileManager()
    ### Training Data ###
    training_data = file_manager.load_csv('data/train.csv')


    for i in range(len(training_data.index)):
        # Representation
        current_vector_data = training_data.loc[i]
        current_vector_data = current_vector_data.values.tolist()
        current_row_vector = RowVector(
            current_vector_data[0], current_vector_data[1], current_vector_data[2:])

        # Model Fitting



    return


if __name__ == "__main__":

    main()
