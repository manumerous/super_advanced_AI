# ___________.__.__              _____
# \_   _____/|__|  |   ____     /     \ _____    ____ _____     ____   ___________
#  |    __)  |  |  | _/ __ \   /  \ /  \\__  \  /    \\__  \   / ___\_/ __ \_  __ \
#  |     \   |  |  |_\  ___/  /    Y    \/ __ \|   |  \/ __ \_/ /_/  >  ___/|  | \/
#  \___  /   |__|____/\___  > \____|__  (____  /___|  (____  /\___  / \___  >__|
#      \/                 \/          \/     \/     \/     \//_____/      \/

""" The file manager loads csv files into pandas dataframes 
and can furthermore move and rename files"""

__author__ = "Manuel Galliker"
__maintainer__ = "Manuel Galliker"
__license__ = "GPL"

import os
import glob
import pandas as pd
import time
from termcolor import colored, cprint
from shutil import copyfile


class FileManager():

    def __init__(self):
        self.working_directory =os.getcwd()


    # loads a csv file if there's exactly one csv file in the folder. 
    # This function is made so you only need to specify the path and
    #  you don't need to know the filename
    def load_single_csv_from_folder(self, relative_path):
        path = os.path.join(self.working_directory, relative_path)
        all_files = glob.glob(os.path.join(path, "*.csv"))
        file_name_list = []
        big_frame = dict()
        for file in all_files:
            # Getting the file name without extension
            file_name = os.path.splitext(os.path.basename(file))[0]
            file_name_list.append(file_name)
            # Reading the file content to create a DataFrame
            dfn = pd.read_csv(file)
            # Setting the file name (without extension) as the index name
            dfn.index.name = file_name
            big_frame[file_name] = dfn
        file_name_list.sort()
        # check whether the folder contains only one file, which is the desired
        # case.
        if len(file_name_list) == 1:
            # convert to list and dict to dataframe and string for single file
            # loaded.
            file_name = file_name_list[0]
            single_frame = big_frame[file_name]
            return single_frame, file_name
        # otherwise give error feedback
        elif len(file_name_list) > 1:
            cprint(("Error loading single file from folder: Multiple files detected in folder!"), "red")
            return
        else:
            cprint(("Error loading single file from folder: No file detected in folder!"), "red")
            return

    # function to load all csv files from a certain folder and add them to a big dataframe
    def load_all_csvs_from_folder(self, relative_path):
        print(self.working_directory)
        path = os.path.join(self.working_directory, relative_path)
        print(path)
        print("Loaded files:")
        all_files = glob.glob(os.path.join(path, "*.csv"))
        file_name_list = []
        big_frame = dict()
        for file in all_files:
            # Getting the file name without extension
            file_name = os.path.splitext(os.path.basename(file))[0]
            file_name_list.append(file_name)
            # Reading the file content to create a DataFrame
            dfn = pd.read_csv(file)
            # Setting the file name (without extension) as the index name
            dfn.index.name = file_name
            big_frame[file_name] = dfn
        file_name_list.sort()
        if len(file_name_list) < 1:
            cprint(("Error loading all files from folder: No file detected in folder!"), "red")
            return
        print("\n {} \n".format(file_name_list))
        return big_frame, file_name_list

    # renames a file given its path, current name and desired new name
    def rename_file(self, relative_path, current_file_name, new_file_name):
        file_dir = os.path.join(self.working_directory, relative_path)
        try:
            old_file_path = os.path.join(file_dir, current_file_name)
            new_file_path = os.path.join(file_dir, new_file_name)
            os.rename(old_file_path, new_file_path)
        except:
            cprint('Could not rename file', 'red')

    # function to move a file
    def move_file(self, relative_source_path, relative_destination_path, file_name):
        try:
            source_path = os.path.join(self.working_directory, relative_source_path)
            destination_path = os.path.join(self.working_directory, relative_destination_path)
            source_file = os.path.join(source_path, file_name)
            destination_file = os.path.join(destination_path, file_name)
            os.rename(source_file, destination_file)
            self.move_file_successful = True
        except:
            cprint('Could not move file to local archive', 'red')

    def copy_file(self, source_path, destination_path, file_name):
        try:
            source_file = os.path.join(source_path, file_name)
            destination_file = os.path.join(destination_path, file_name)
            copyfile(source_file, destination_file)
            self.copy_file_successful = True
        except:
            cprint('Could not copy file', 'red')