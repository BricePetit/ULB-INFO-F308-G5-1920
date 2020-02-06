from os import listdir
from os.path import isfile, isdir, join
#Loading data

class Parser(object):

    def __init__(self,path,size = 0.8):

        self.path = path
        self.dir_names = self.getDirectories()
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
            nb = int(len(fnames) * size)
            self.train_files[category] = fnames[:nb]
            self.test_files[category] = fnames[nb:]
