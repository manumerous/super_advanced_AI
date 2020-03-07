__author__ = "Manuel Galliker"
__maintainer__ = "Manuel Galliker"
__license__ = "GPL"

class RowVector():

    def __init__(self, vector_index, vector_mean, raw_vector_data):
        self.id = vector_index
        self.mean = vector_mean
        self.mean_hat = 'none'
        self.data = raw_vector_data
