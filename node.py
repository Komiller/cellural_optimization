import numpy as np


class Node(object):
    def __init__(self, name, x, y, z):
        self.__name = name
        self.__x = x
        self.__y = y
        self.__z = z

    def get_coordinate_array(self):
        return np.array([self.__x, self.__y, self.__z])
