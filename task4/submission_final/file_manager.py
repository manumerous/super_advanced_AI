'''The main file is used to start up the program'''

__author__ = "Manuel Galliker"
__maintainer__ = "Manuel Galliker"
__license__ = "GPL"

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input
from os import listdir
from os import path
from os import makedirs
import sys
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import Sequence
from math import floor
from math import ceil

class TrainBatchProvider(Sequence):
    """
    This class allowes the model to acces data as it needs it. 
    """

    def __init__(self, batch_size=16):
        self.batch_size = batch_size
        self.training_data = np.loadtxt("train_triplets.txt", dtype=np.int32, delimiter=" ")
        self.length = floor(self.training_data.shape[0]/self.batch_size)
        self.on_epoch_end()

    def __getitem__(self, index):
        triplets = self.training_data[self.batch_size*index:self.batch_size*(index+1), :]
        A = preprocess_input(np.asarray([img_to_array(load_img(covert_to_path(t[0]))) for t in triplets]))
        P = preprocess_input(np.asarray([img_to_array(load_img(covert_to_path(t[1]))) for t in triplets]))
        N = preprocess_input(np.asarray([img_to_array(load_img(covert_to_path(t[2]))) for t in triplets]))

        # labels for keras
        labels = np.asarray([1]*self.batch_size, dtype=np.float32)
        return [A, P, N], labels

    def __len__(self):
        return self.length

def covert_to_path(i):
    #return the corresponding path to image i
    return "resized_images/{:0>5}.jpg".format(i)


def resize_images(sizex, sizey):
    
    pics = listdir("food")
    picture_list = [pic for pic in pics if pic.endswith(".jpg")]
    for pic in tqdm(picture_list):
        with Image.open("food/" + pic) as img:
            res = img.resize((sizex, sizey), Image.LANCZOS)
            res.save("resized_images/" + pic)


def load_images(image_triplet_list): 
    # Open the image form working directory
    filename_A = 'resized_images/' + str(image_triplet_list[0]) + '.jpg'
    image_A = Image.open(filename_A)
    filename_B = 'resized_images/' + str(image_triplet_list[1]) + '.jpg'
    image_B = Image.open(filename_B)
    filename_C = 'resized_images/' + str(image_triplet_list[2]) + '.jpg'
    image_C = Image.open(filename_C)
    img_A_array = np.asarray(image_A)
    img_B_array = np.asarray(image_B)
    img_C_array = np.asarray(image_C)
    return img_A_array, img_B_array, img_C_array


def return_training_data(batch_size, batch_counter, image_size):
    all_train_triplets = np.genfromtxt("train_triplets.txt",dtype='str')
    if (batch_size*(batch_counter+1) > all_train_triplets.shape[0]):
        batch_size = all_train_triplets.shape[0] -batch_size*batch_counter

    img_A_array = np.zeros((batch_size, image_size[0], image_size[1], image_size[2]))
    img_P_array = np.zeros((batch_size, image_size[0], image_size[1], image_size[2]))
    img_N_array = np.zeros((batch_size, image_size[0], image_size[1], image_size[2]))

    for i in range(batch_size):
        cur_img_A_array, cur_img_P_array, cur_img_N_array = load_images(all_train_triplets[i+ batch_size*batch_counter])
        img_A_array[i]= cur_img_A_array
        img_P_array[i]= cur_img_P_array
        img_N_array[i]= cur_img_N_array

    A = preprocess_input(img_A_array)
    P = preprocess_input(img_P_array)
    N = preprocess_input(img_N_array)
    return [A, P, N]

def return_test_data(batch_size, image_size, batch_counter):
    all_test_triplets = np.genfromtxt("test_triplets.txt",dtype='str')
    if (batch_size*(batch_counter+1) > all_test_triplets.shape[0]):
        batch_size = all_test_triplets.shape[0] -batch_size*batch_counter

    img_A_array = np.zeros((batch_size, image_size[0], image_size[1], image_size[2]))
    img_P_array = np.zeros((batch_size, image_size[0], image_size[1], image_size[2]))
    img_N_array = np.zeros((batch_size, image_size[0], image_size[1], image_size[2]))
    for i in range(batch_size):
        cur_img_A_array, cur_img_P_array, cur_img_N_array = load_images(all_test_triplets[i + batch_size*batch_counter])
        img_A_array[i]= cur_img_A_array
        img_P_array[i]= cur_img_P_array
        img_N_array[i]= cur_img_N_array

    A = preprocess_input(img_A_array)
    P = preprocess_input(img_P_array)
    N = preprocess_input(img_N_array)
    return [A, P, N]


     





