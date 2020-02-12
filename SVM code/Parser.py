from os import listdir
from os.path import isfile, isdir, join
from sklearn.model_selection import train_test_split

class Parser(object):

    def __init__(self,path,size = 0.8):

        self.path = path
        self.dir_names = self.getDirectories()
        self.files = {}
        self.train_files = {}
        self.test_files = {}
        self.generate_dataset(size)

    def getFiles(self,directory):

        files_list = []
        for f in listdir(self.path + directory + "/"):
            if isfile(join(self.path + directory + "/", f)):
                files_list.append(join(self.path + directory + "/", f))

        return files_list

    def getDirectories(self):

        return [d for d in listdir(self.path) if isdir(join(self.path, d))]

    def getClassNames(self):

        return [d for d in self.dir_names]

    def getTrainingFiles(self):

        return self.train_files

    def getTestingFiles(self):

        return self.test_files

    def generate_dataset(self,size):

        for category in self.dir_names:
            fnames = self.getFiles(category)
            self.files[category] = fnames

    def dataset_split(self,size = 0.2):
        for category in self.dir_names:
            self.train_files[category], self.test_files[category] = train_test_split(self.files[category], test_size = size)
