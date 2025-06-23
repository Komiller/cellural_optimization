class Strut(object):
    def __init__(self, name, start_node, end_node, material, radius):
        self.__name = name
        self.__start_node = start_node
        self.__end_node = end_node
        self.__material = material
        self.__radius = radius

    def get_start_node(self):
        return self.__start_node

    def get_end_node(self):
        return self.__end_node

    def get_material(self):
        return self.__material

    def get_radius(self):
        return self.__radius
