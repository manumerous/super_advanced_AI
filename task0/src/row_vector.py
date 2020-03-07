__author__ = "Manuel Galliker"
__maintainer__ = "Manuel Galliker"
__license__ = "GPL"

class RowVector():

    def __init__(self, vector_index, raw_vector_data):
        self.id = vector_index
        self.mean_value = 'none'
        self.data = raw_vector_data
