class Material:
    def __init__(self, name, modulus_of_elasticity, poissons_ratio, thermal_conductivity):
        self.__name = name
        self.__modulus_of_elasticity = modulus_of_elasticity
        self.__poissons_ratio = poissons_ratio
        self.__thermal_conductivity = thermal_conductivity
        self.__contained_materials = tuple([self.__name])

    def get_name(self):
        return self.__name

    def get_modulus_of_elasticity(self):
        return self.__modulus_of_elasticity

    def get_poissons_ratio(self):
        return self.__poissons_ratio

    def get_thermal_conductivity(self):
        return self.__thermal_conductivity

    def get_contains(self):
        return self.__contained_materials

    def set_contains(self, contained_materials):
        self.__contained_materials = contained_materials
        return self
