from os import listdir
from os.path import isfile, isdir, join

from sklearn.model_selection import train_test_split


class Parser(object):

    def __init__(self, path):

        self.path = path
        self.dir_names = self.get_directories()
        self.files = {}
        self.train_files = {}
        self.test_files = {}
        self.generate_dataset()

    def get_files(self, directory):

        files_list = []
        for f in listdir(self.path + directory + "/"):
            if isfile(join(self.path + directory + "/", f)):
                files_list.append(join(self.path + directory + "/", f))

        return files_list

    def get_directories(self):

        return [d for d in listdir(self.path) if isdir(join(self.path, d))]

    def generate_dataset(self):

        for category in self.dir_names:
            files = self.get_files(category)
            self.files[category] = files

    def dataset_split(self, size=0.2):
        for category in self.dir_names:
            self.train_files[category], self.test_files[category] = \
                train_test_split(self.files[category],
                                 test_size=size,
                                 random_state=42)
