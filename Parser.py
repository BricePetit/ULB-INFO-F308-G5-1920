from os import listdir
from os.path import isfile, join
#Loading data

class Parser(object):

    def __init__(self,path,size = 2/3):

        self.path = path
        self.dir_names = self.getDirectories()
        self.train_files = {}
        self.test_files = []
        self.generate_dataset(size)

    def getFiles(self):

        return [f for f in listdir(self.path) if isfile(join(self.path, f))]

    def getDirectories(self):

        return [d for d in listdir(self.path) if not isfile(join(self.path, d))]

    def getClassNames(self):

        return [d for d in self.dir_names]

    def getTrainingFiles(self):

        return self.train_files

    def getTestingFiles(self):

        return self.test_files

    def generate_dataset(self,size):

        class_num = 0
        for d in self.dir_names:
            fnames = [f for f in listdir(self.path+d+"/") if isfile(join(self.path+d+"/", f))]
            #names = [join(self.path+d+"/", f) for f in listdir(self.path+d+"/") if isfile(join(self.path+d+"/", f))]
            nb = int(len(fnames) * size)
            self.train_files[(d, class_num, self.path+d+"/")] = fnames[:nb]
            self.test_files += [(d,join(self.path+d+"/", f)) for f in fnames[nb:]]
            class_num += 1
