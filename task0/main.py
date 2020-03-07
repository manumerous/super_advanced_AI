__author__ = "Manuel Galliker"
__maintainer__ = "Manuel Galliker"
__license__ = "GPL"

from src import FileManager

def main():
    print('test')
    file_manager = FileManager()
    big_frame = file_manager.load_all_csvs_from_folder('data/')
    print(big_frame)
    return


if __name__ == "__main__":

    main()
