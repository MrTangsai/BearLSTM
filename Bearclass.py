import scipy.io as scio
import numpy as np
import os
import re

n_point = 200
n_num = 2400


class Bear(object):
    """docstring for Bear"""

    def __init__(self):
        super(Bear, self).__init__()
        self.data = self.get_data()
        self.target = self.get_target()

    def get_File(self):
        return os.listdir('data/')

    def get_data(self):
        Filelist = self.get_File()
        for i in range(len(Filelist)):
            Fileres = scio.loadmat('data/{}'.format(Filelist[i]))
            for x in Fileres.keys():
                keys = re.match('X\d{3}_DE_time', x)
                if keys:
                    key = keys.group()
            assert len(Fileres[key]) >= 480000
            #print(key, ...)
            if i == 0:
                data = np.array(Fileres[key][0:480000]
                                ).reshape((n_num, n_point))
            else:
                data = np.vstack(
                    (data, np.array(Fileres[key][0:480000]).reshape((n_num, n_point))))
            # print(data.shape)
        return data

    def get_target(self):
        Filelist = self.get_File()
        val = list(map(lambda x: x.replace('.mat', ''), Filelist))
        val = np.array(val)[:, np.newaxis]
        target = np.copy(val)
        for _ in range(n_num - 1):
            target = np.hstack((target, val))
        return target.flatten()

if __name__ == '__main__':
    bear = Bear()
    print(bear.data.shape, ...)
    print(bear.target.shape, ...)
    #print(bear.get_File(), ...)
