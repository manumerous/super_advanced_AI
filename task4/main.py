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

import numpy as np
import pandas as pd
from PIL import Image

def load_triple_set(): 
    triple_matrix = np.loadtxt(fname = "test_triplets.txt")
    print(triple_matrix)
    return

def load_images(image_triplet_list): 
    print(image_triplet_list)
    # Open the image form working directory
    filename_A = 'food/' + str(image_triplet_list[0]) + '.jpg'
    image_A = Image.open(filename_A)
    filename_B = 'food/' + str(image_triplet_list[1]) + '.jpg'
    image_B = Image.open(filename_A)
    filename_C = 'food/' + str(image_triplet_list[2]) + '.jpg'
    image_C = Image.open(filename_A)
    # summarize some details about the image
    print(image_A.format)
    print(image_A.size)
    print(image_A.mode)
    # show the image
    image_A.show()
    img_A_array = np.asarray(image_A)
    print(img_A_array)
    print(img_A_array.shape)




def main():
    triplet_matrix = np.genfromtxt("train_triplets.txt",dtype='str')
    load_images(triplet_matrix[0])

    return



if __name__ == "__main__":

    main()