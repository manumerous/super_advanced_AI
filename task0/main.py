__author__ = "Manuel Galliker"
__maintainer__ = "Manuel Galliker"
__license__ = "GPL"

from src import FileManager
from src import RowVector
from src import rsme_function
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
        current_row_vector = RowVector(current_vector_data[0], current_vector_data[1], current_vector_data[2:])
        
        # Model Fitting
        mean_collector.append(current_row_vector.mean)
        mean_hat_collector.append(statistics.mean(current_row_vector.data))

    # Evaluation by rsmi based on training set
    rsme_error = rsme_function(mean_collector, mean_hat_collector)
    print('the root mean square error is:')
    print(rsme_error)

    return

    ### Predicition ### 


if __name__ == "__main__":

    main()
