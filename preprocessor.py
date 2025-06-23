import os

import numpy as np
import node
import material as m
import strut
from icecream import ic

def read_input_file(filename):
    input_file = open(filename, "r")
    lines = input_file.readlines()

    node_lookup_table = {}
    strut_lookup_table = {}
    material_lookup_table = {}
    data = [x.split() for x in lines]

    for line in data:
        if len(line) > 0:
            if (line[0]).lower() == "node":
                node_lookup_table[int(line[1])] = node.Node(int(line[1]), float(line[2]), float(line[3]),
                                                            float(line[4]))
            if (line[0]).lower() == "mat":
                new_material = m.Material(int(line[1]), float(line[2]), float(line[3]), float(line[4]))
                material_lookup_table[new_material.get_contains()] = new_material
    for line in data:
        if len(line) > 0:
            if (line[0]).lower() == "strut":
                strut_lookup_table[int(line[1])] = strut.Strut(int(line[1]), node_lookup_table[int(line[2])],
                                                               node_lookup_table[int(line[3])],
                                                               material_lookup_table[tuple([int(line[4])])],
                                                               float(line[5]))


    input_file.close()

    return node_lookup_table, strut_lookup_table, material_lookup_table


def generate_voxel(n, filename,radises):
    size = 1 / n
    voxel = np.zeros((n, n, n))
    voxel_centers=np.zeros((n,n,n,6))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                voxel_centers[i][j][k][0] = k
                voxel_centers[i][j][k][1] = i
                voxel_centers[i][j][k][2] = j
                voxel_centers[i][j][k][3] = (i + 0.5) * size
                voxel_centers[i][j][k][4] = (j + 0.5) * size
                voxel_centers[i][j][k][5] = (k + 0.5) * size

    voxel_centers=voxel_centers.reshape((n**3,6))



    nodes, struts, materials = read_input_file(filename)
    np.seterr(invalid='ignore')
    contained=np.zeros((len(voxel_centers),len(struts)),dtype=np.int8)-1
    for i,key in enumerate(struts):

        strut = struts[key]
        voxel_center = voxel_centers[:, 3:6]
        distance = calculate_distance(voxel_center, strut.get_start_node(), strut.get_end_node())
        radius = radises[key-1]
        mask=np.logical_or(distance < radius,np.isclose(distance, radius, 1.e-12, 1.e-12))
        contained[mask,i]=strut.get_material().get_name()

    for j in range(len(struts)):
        mask1 = contained[:,j]!=-1
        for i in range(len(contained)):
            if mask1[i]:
                voxel[int(voxel_centers[i, 0]), int(voxel_centers[i, 1]), int(voxel_centers[i, 2])] = materials[
                    (contained[i, j],)].get_name()
            else:
                continue



    """
    mask1=np.all(contained[:,0::2]!=-1*np.ones(2))
    mask = voxel_centers[mask1][0::3]
    for i in range(len(contained)):
        if mask1[i]:
            voxel[voxel_centers[i, 0], voxel_centers[i, 2], voxel_centers[i, 3]] = get_new_mat(contained[i])
        else:
            continue"""


    density = calculate_density(voxel, n)
    lambdas, mus, thermals = characterize(materials)
    return voxel, density, lambdas, mus, thermals



def calculate_distance(voxel_center, start_node, end_node):
    start_node = start_node.get_coordinate_array()
    end_node = end_node.get_coordinate_array()
    alpha = np.rad2deg(
        np.arccos(np.clip((np.dot((voxel_center - start_node) / np.linalg.norm(voxel_center - start_node),
                                  (end_node - start_node).T / np.linalg.norm(end_node - start_node))), -1., 1.)))
    beta = np.rad2deg(np.arccos(np.clip((np.dot((voxel_center - end_node) / np.linalg.norm(voxel_center - end_node),
                                                (start_node - end_node).T) / np.linalg.norm(start_node - end_node)),
                                        -1., 1.)))

    distances = np.zeros(len(voxel_center), dtype=np.float64)
    mask = np.logical_and(alpha < 90, beta < 90)
    if len(distances[mask]) > 0:
        distances[mask] = np.linalg.norm(np.cross(end_node - start_node, voxel_center[mask] - start_node), axis=1,
                                         ord=np.inf) / np.linalg.norm(
            end_node - start_node)
    mask = (alpha >= 90)
    if len(distances[mask]) > 0:
        distances[mask] = np.linalg.norm(voxel_center[mask] - start_node, axis=1, ord=np.inf)
    mask = (beta >= 90)
    if len(distances[mask]) > 0:
        distances[mask] = np.linalg.norm(voxel_center[mask] - end_node, axis=1, ord=np.inf)
    return distances




def create_a_new_material(materials, contained_materials):
    modulus_of_elasticity = 0
    poissons_ratio = 0
    thermal_conductivity = 0

    for index in contained_materials:
        material = tuple([index])
        modulus_of_elasticity += materials[material].get_modulus_of_elasticity()
        poissons_ratio += materials[material].get_poissons_ratio()
        thermal_conductivity += materials[material].get_thermal_conductivity()
    modulus_of_elasticity = modulus_of_elasticity / len(contained_materials)
    poissons_ratio = poissons_ratio / len(contained_materials)
    thermal_conductivity = thermal_conductivity / len(contained_materials)


    materials[contained_materials] = m.Material(len(materials) + 1, modulus_of_elasticity, poissons_ratio,
                                                thermal_conductivity)

    return materials


def characterize(material_list):
    lambdas = []
    mus = []
    thermals = []
    for index in material_list:
        material = material_list[index]
        lambdas.append(get_lames_first_parameter(material.get_modulus_of_elasticity(), material.get_poissons_ratio()))
        mus.append(get_lames_second_parameter(material.get_modulus_of_elasticity(), material.get_poissons_ratio()))
        thermals.append(material.get_thermal_conductivity())
    return lambdas, mus, thermals


def calculate_density(voxel, n):
    return np.count_nonzero(voxel) * n ** (-3)


def get_lames_first_parameter(youngs_modulus, poissons_ratio):
    return youngs_modulus * poissons_ratio / ((1 + poissons_ratio) * (1 - 2 * poissons_ratio))


def get_lames_second_parameter(youngs_modulus, poissons_ratio):
    return youngs_modulus / (2 * (1 + poissons_ratio))



def get_new_mat(a):
    global materials
    mats=[]
    for mat in a:
        if mat != -1:mats.append(mat)
    contained_materials = tuple([*set(mats)])
    materials = create_a_new_material
    return materials[contained_materials].get_name()


