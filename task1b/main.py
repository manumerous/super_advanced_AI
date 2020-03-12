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
    mean_collector = []
    mean_hat_collector = []

    for i in range(len(training_data.index)):
        # Representation
        current_vector_data = training_data.loc[i]
        current_vector_data = current_vector_data.values.tolist()
        current_row_vector = RowVector(
            current_vector_data[0], current_vector_data[1], current_vector_data[2:])

        # Model Fitting
        mean_collector.append(current_row_vector.mean)
        mean_hat_collector.append(statistics.mean(current_row_vector.data))

    # Evaluation by rsmi based on training set
    rsme_error = rsme_function.rsme_function(
        mean_collector, mean_hat_collector)
    print('the root mean square error is:')
    print(rsme_error)

    ### Test Data ###
    test_data = file_manager.load_csv('data/test.csv')
    id_collector = []
    mean_hat_collector = []

    for i in range(len(test_data.index)):
        # Representation
        current_vector_data = test_data.loc[i]
        current_vector_data = current_vector_data.values.tolist()
        current_row_vector = RowVector(
            current_vector_data[0], 0, current_vector_data[1:])

        # Model Fitting
        id_collector.append(current_row_vector.id)
        mean_hat_collector.append(statistics.mean(current_row_vector.data))

    output_df = pd.DataFrame(
        list(zip(id_collector, mean_hat_collector)), columns=['Id', 'y'])
    file_manager.save_dataframe_to_csv(output_df, 'data/submission_file.csv')

    return


if __name__ == "__main__":

    main()
