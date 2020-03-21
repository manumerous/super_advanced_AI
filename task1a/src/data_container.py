# ________          __           _________                __         .__                     
# \______ \ _____ _/  |______    \_   ___ \  ____   _____/  |______  |__| ____   ___________ 
#  |    |  \\__  \\   __\__  \   /    \  \/ /  _ \ /    \   __\__  \ |  |/    \_/ __ \_  __ \
#  |    `   \/ __ \|  |  / __ \_ \     \___(  <_> )   |  \  |  / __ \|  |   |  \  ___/|  | \/
# /_______  (____  /__| (____  /  \______  /\____/|___|  /__| (____  /__|___|  /\___  >__|   
#         \/     \/          \/          \/            \/          \/        \/     \/       

'''The DataContainer Object was used to ensure the correct data representation'''

__author__ = "Manuel Galliker"
__maintainer__ = "Manuel Galliker"
__license__ = "GPL"


class DataContainer():

    def __init__(self, data_matrix):
        self._parse_row_vector_matrix(data_matrix)

    def _parse_row_vector_matrix(self, data_matrix):
        self.id = data_matrix[:, 0]
        self.y = data_matrix[:, 1]
        self.y_hat = 'none'
        self.x = data_matrix[:, 2:]

    def get_id(self):
        return self.id

    def get_y(self):
        return self.y

    def get_y_hat(self):
        return self.y_hat

    def get_x(self):
        return self.x

    def set_y_hat(self, value):
        self.y_hat = value
