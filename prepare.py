import time

import numpy as np
import os
from itertools import permutations

from scipy.spatial import KDTree
def load_CHs():
    folder_path='PA_massive'
    matrices = {}
    count=0
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.npy'):
            file_path = os.path.join(folder_path, file_name)
            # Загружаем матрицу из файла
            matrix = np.load(file_path)
            # Сохраняем в словарь без расширения
            name=os.path.splitext(file_name)[0]
            parts=name.split('_')
            numbers =tuple([float(part) for part in parts])
            matrices[count] = (numbers,matrix)
            count+=1
    return matrices
#def add_rotation
def load_dens():
    folder_path='PA_densities'
    matrices = {}
    count=0
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.npy'):
            file_path = os.path.join(folder_path, file_name)
            # Загружаем матрицу из файла
            matrix = np.load(file_path)
            # Сохраняем в словарь без расширения
            name=os.path.splitext(file_name)[0]
            parts=name.split('_')
            numbers =tuple([float(part) for part in parts])
            matrices[count] = (numbers,matrix)
            count+=1
    return matrices



def rotation_matrix_init(axis,theta):
    if axis == 'X':
        Ro=np.array([[1, 0, 0],
                     [0,np.cos(theta),-np.sin(theta)],
                     [0,np.sin(theta),np.cos(theta)]])
    elif axis == 'Y':
        Ro=np.array([[np.cos(theta), 0, np.sin(theta)],
                     [0,1,0],
                     [-np.sin(theta),0,np.cos(theta)]])
    else:
        Ro=np.array([[np.cos(theta),-np.sin(theta), 0],
                     [np.sin(theta),np.cos(theta),0],
                     [0,0,1]])

    l1,l2,l3,m1,m2,m3,n1,n2,n3=tuple(Ro.flatten())
    """R=np.array([[l1*l1,m1*m1,n1*n1,2*m1*n1,2*n1*l1,2*l1*m1],
                [l2*l2,m2*m2,n2*n2,2*m2*n2,2*n2*l2,2*l2*m2],
                [l3*l3,m3*m3,n3*n3,2*m3*n3,2*n3*l3,2*l3*m3],
                [l3 * l1, m3 * m1, n3 * n1, m3 * n1 + m1 * n3, n3 * l1 + n1 * l3, l3 * m1 + l1 * m3],
                [l1*l2,m1*m2,n1*n2,m1*n2+m2*n1,n1*l2+n2*l1,l1*m2+l2*m1],
                [l2*l3,m2*m3,n2*n3,m2*n3+m3*n2,n2*l3+n3*l2,l2*m3+l3*m2]])"""
    R = np.array([[l1 * l1, l2 * l2, l3 * l3, l1 * l2, l2 * l3, l1 * l3],
                  [m1 * m1, m2 * m2, m3 * m3, m1 * m2, m2 * m3, m1 * m3],
                  [n1 * n1, n2 * n2, n3 * n3, n1 * n2, n2 * n3, n1 * n3],
                  [l1 * m1, l2 * m2, l3 * m3, l1 * m2 + l2 * m1, l2 * m3 + l3 * m2, l1 * m3 + l3 * m1],
                  [m1 * n1, m2 * n2, m3 * n3, m1 * n2 + m2 * n1, m2 * n3 + m3 * n2, m1 * n3 + m3 * n1],
                  [l1 * n1, l2 * n2, l3 * n3, l1 * n2 + l2 * n1, l2 * n3 + l3 * n2, l1 * n3 + l3 * n1]])
    return R

def rotate(matrix,axis,theta,flag=0):
    R=rotation_matrix_init(axis,theta)
    if flag==1:
        ic(R)
    return R.dot(matrix).dot(R.T)





from icecream import ic
def make_full_massive(massive):
    full=[]
    for var in massive:
        flag=0
        point = massive[var][0]
        matrix = massive[var][1]
        full.append(massive[var])
        if point[0]==point[1] and point[0]==point[2]:
            continue

        if point[1]!=point[2]:
            pointx = [point[0], point[2], point[1]]
            matrixx = rotate(matrix, 'X', np.pi/2)

            full.append((pointx, matrixx))

        if point[0]!=point[2]:
            pointy = [point[2], point[1], point[0]]
            matrixy= rotate(matrix, 'Y', np.pi/2)
            full.append((pointy,matrixy))


        if point[0]!=point[1]:
            pointz = [point[1], point[0], point[2]]
            matrixz = rotate(matrix, 'Z', np.pi/2)
            full.append((pointz, matrixz))

            if  point[1]!=point[2]:
                if pointx[0] != pointx[1]:
                    pointxz = [point[1], point[2], point[0]]
                    matrixxz = rotate(matrixy, 'Z', np.pi/2)
                    full.append((pointxz, matrixxz))

            if  point[0]!=point[2]:
                if pointy[0] != pointy[1]:
                    pointyz = [point[2], point[0], point[1]]
                    matrixyz = rotate(matrixx, 'Z', np.pi/2)
                    full.append((pointyz, matrixyz))


    return full


"""folder_path='new_PA'
matrices = {}
count=0
for file_name in os.listdir(folder_path):
    if file_name.endswith('.npy'):
        file_path = os.path.join(folder_path, file_name)
        # Загружаем матрицу из файла
        matrix = np.load(file_path)
        # Сохраняем в словарь без расширения
        name=os.path.splitext(file_name)[0]
        parts=name.split('_')
        numbers =tuple([float(part) for part in parts])
        if (round(numbers[0],4),round(numbers[1],4),round(numbers[2],4))==(0.3,0.4,0.5):
            print(1)
            break
matr={0:(numbers,matrix)}
np.set_printoptions(threshold=np.inf,linewidth=np.inf,suppress=True)
tmp=make_full_massive(matr)
for var in tmp:
    print(var[0])
    print(np.round(var[1],0))"""

