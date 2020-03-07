__author__ = "Manuel Galliker"
__maintainer__ = "Manuel Galliker"
__license__ = "GPL"

from src import FileManager
from src import RowVector
import statistics

def main():
    print('test')
    file_manager = FileManager()
    big_frame = file_manager.load_all_csvs_from_folder('data/')
    training_data = file_manager.load_csv('data/train.csv')
    # print(training_data)

    for i in range(len(training_data.index)):
        current_vector_data = training_data.loc[i]
        current_vector_data = current_vector_data.values.tolist()
        current_row_vector = RowVector(current_vector_data[0], current_vector_data[2:])
        current_row_vector.mean_value = statistics.mean(current_row_vector.data)

    return


if __name__ == "__main__":

    main()
