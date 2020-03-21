# __________              ____   ____             __
# \______   \ ______  _  _\   \ /   /____   _____/  |_  ___________
#  |       _//  _ \ \/ \/ /\   Y   // __ \_/ ___\   __\/  _ \_  __ \
#  |    |   (  <_> )     /  \     /\  ___/\  \___|  | (  <_> )  | \/
#  |____|_  /\____/ \/\_/    \___/  \___  >\___  >__|  \____/|__|
#         \/                            \/     \/

'''The RowVector Object was used to ensure the correct data representation for task 0'''

__author__ = "Manuel Galliker"
__maintainer__ = "Manuel Galliker"
__license__ = "GPL"


class RowVector():

    def __init__(self, vector_index, vector_mean, raw_vector_data):
        self.id = vector_index
        self.mean = vector_mean
        self.mean_hat = 'none'
        self.data = raw_vector_data
