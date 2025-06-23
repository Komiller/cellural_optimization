import time
import os
import pythoncom
import PyFemap
import sys
import win32com.client.gencache
import numpy as np
from preprocessor import generate_voxel
from PBC_femap import apply_3d_PBC

from femap_integration import create_matirial_iso,create_property
#from model import model

try:
    win32com.client.gencache.is_readonly=False
    existObj = pythoncom.connect(PyFemap.model.CLSID) #Grabs active model
    app = PyFemap.model(existObj)#Initializes object to active mode
except:
    sys.exit("femap not open") #Exits program if there is no active femap model

def create_model(radii,nvoxel):
    x,y,z=0,0,0
    nodes = {}
    elements = []
    node_id = 2
    h_voxel = 1 / nvoxel
    voxel, density, lambdas, mus, thermals = generate_voxel(nvoxel, 'PA.txt',radii)
    for vx in range(nvoxel):
        for vy in range(nvoxel):
            for vz in range(nvoxel):
                if voxel[vz, vx, vy] == 0: continue
                elm_nodes = np.array([(x + vx * h_voxel, y + (vy + 1) * h_voxel, z + vz * h_voxel),
                                      (x + vx * h_voxel, y + (vy + 1) * h_voxel, z + (vz + 1) * h_voxel),
                                      (x + vx * h_voxel, y + vy * h_voxel, z + (vz + 1) * h_voxel),
                                      (x + vx * h_voxel, y + vy * h_voxel, z + vz * h_voxel),
                                      (x + (vx + 1) * h_voxel, y + (vy + 1) * h_voxel, z + vz * h_voxel),
                                      (x + (vx + 1) * h_voxel, y + (vy + 1) * h_voxel, z + (vz + 1) * h_voxel),
                                      (x + (vx + 1) * h_voxel, y + vy * h_voxel, z + (vz + 1) * h_voxel),
                                      (x + (vx + 1) * h_voxel, y + vy * h_voxel, z + vz * h_voxel)])
                elm_nodes = np.round(elm_nodes, 4)
                for node in elm_nodes:
                    nid = nodes.get(tuple(node))
                    if nid is None:
                        nodes[tuple(node)] = node_id
                        nid = node_id
                        node_id += 1
                    elements.append(nid)
                for _ in range(12): elements.append(0)

    xyz = []
    for key in nodes:
        xyz.append(key[0])
        xyz.append(key[1])
        xyz.append(key[2])
    ids = np.arange(2, node_id)
    node = app.feNode
    node.PutCoordArray(len(ids), ids, xyz)

    elem = app.feElem
    nel = len(elements) // 20
    elem.PutAllArray(nel, np.arange(1, nel + 1), [1] * nel, [25] * nel, [8] * nel, [1] * nel, [124] * nel,
                           [0] * 2 * nel, [0] * 3 * nel, [0] * nel * 6, [0] * nel * 12, [0] * nel, [0] * nel, elements,
                           [0] * 2 * nel, [0] * 2 * nel)

    #creating BC nodal
    bcset = app.feBCSet
    bcset.title = 'first'
    setID = 1
    bcset.Put(setID)

    bcd = app.feBCDefinition
    bcd.SetID = setID
    bcd.OnType = 7
    bcd.DataType = 18
    bcd.title = "MyDef"
    bcd.Put(1)

    myset = app.feSet
    myset.Add(nodes.get((0.5,0.5,0.5)))


    bc = app.feBCNode
    bc.color = 115
    bc.layer = 1
    bc.expanded = False
    bc.SetID = 1
    bc.BCDefinitionID = 1
    bc.NonZeroConstraint = False
    bc.vdof = (True, True, True, True, True, True)
    bc.Add(myset.ID, True, True, True, True, True, True)
    return len(ids),nel


load_vdofs={0:[1,0,0,0,0],1:[0,1,0,0,0],2:[0,0,1,0,0],3:[1,0,0,0,0],4:[0,1,0,0,0],5:[0,0,1,0,0]}
def create_key_load():
    ls=app.feLoadSet
    ld=app.feLoadDefinition
    le=app.feLoadMesh
    myset = app.feSet
    myset.Add(1)
    for i in range(1,7):
        ls.title = f'F{i}'
        setID = i
        ls.Put(setID)

        ld.SetID = setID
        # if i<3:ld.LoadType = PyFemap.constants.FLT_NFORCE
        # else:ld.LoadType = PyFemap.constants.FLT_NMOMENT
        ld.DataType = PyFemap.constants.FT_NODE
        ld.title = f'F{i}'
        ld.Put(1)

        le.meshID=1
        le.vdof=load_vdofs[i-1]
        le.layer=1
        le.vload=[1]
        le.SetID = setID
        le.LoadDefinitionID=1
        if i-1<3: le.Add(myset.ID,PyFemap.constants.FLT_NFORCE,0,(1,1,1),load_vdofs[i-1],[0,0,0,0,0])
        else: le.Add(myset.ID,PyFemap.constants.FLT_NMOMENT,0,(1,1,1),load_vdofs[i-1],[0,0,0,0,0])

def delet_model(nnl,nel):
    bcset=app.feSet
    bcset.AddRange(1, 2, 1)
    app.feDelete(17, bcset.ID)

    elset = app.feSet
    elset.AddRange(1, nel, 1)
    app.feDeleteMesh(8, elset.ID,False)

    nset = app.feSet
    nset.AddRange(2, nnl+1, 1)
    app.feDelete(7, nset.ID)

    outset=app.feSet
    outset.AddRange(1, 10, 1)
    app.feDelete(28, outset.ID)


def check_last_line_startswith_large(filename, target_string):
    if not os.path.isfile(filename):
        return False
    # Открываем файл в бинарном режиме для эффективного чтения с конца
    with open(filename, 'rb') as file:
        # Перемещаемся в конец файла
        file.seek(0, 2)
        file_size = file.tell()

        # Читаем файл с конца, пока не найдём начало последней строки
        position = file_size - 1
        last_line = []

        while position >= 0:
            file.seek(position)
            char = file.read(1)
            if char == b'\n':
                if position != file_size - 1:  # Игнорируем последний перевод строки
                    break
            else:
                last_line.append(char.decode('utf-8'))
            position -= 1

        # Разворачиваем строку, так как мы читали её с конца
        last_line = ''.join(reversed(last_line))

    target_length = len(target_string)
    return last_line[:target_length] == target_string


import os
import shutil
import re


def copy_with_new_int(directory, old_int, new_int):
    """
    Копирует файлы вида name-XXXX.type, заменяя old_int на new_int.

    :param directory: Директория с файлами
    :param old_int: Исходное число (например, 1234)
    :param new_int: Новое число (например, 5678)
    """
    # Проверяем, что new_int — 4-значное число
    if not (0 <= new_int <= 9999):
        raise ValueError("new_int должен быть в диапазоне 0000..9999")

    # Шаблон для поиска файлов: name-XXXX.type
    pattern = re.compile(r'^(.*?)-(\d{4})(\..+)$')

    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            name_part, int_part, ext_part = match.groups()
            current_int = int(int_part)

            if current_int == old_int:
                # Формируем новое имя: name-new_int.type
                new_filename = f"{name_part}-{new_int:04d}{ext_part}"
                src_path = os.path.join(directory, filename)
                dst_path = os.path.join(directory, new_filename)

                if not os.path.exists(dst_path):
                    shutil.copy2(src_path, dst_path)
                    print(f"Скопировано: {filename} -> {new_filename}")
                else:
                    print(f"Файл {new_filename} уже существует, пропускаем.")




if __name__ == '__main__':

    count=104
    # from itertools import combinations_with_replacement
    # combinations = list(combinations_with_replacement(range(1, 10), 3))
    # result = [list(comb) for comb in combinations]
    # for i in range(count,len(result)):
    #     t1 = time.time()
    #     comb = result[i]
    #     comb = np.array(comb)
    radiuses = [0.2,0.2,0.2]

    # if 0.5 in radiuses:
    #     copy_with_new_int('E:/key_dof_homo', 74, count)
    #     print(f'it = {count}, R = f{radiuses}, es-time-left = {(time.time() - t1) * (len(result) - count)}')
    #     count += 1
    #     continue
    app.feDeleteAll(True, True, True, True)
    create_matirial_iso(200, 0.3)
    create_property()

    node = app.feNode
    node.PutCoordArray(1, 1, [-0.5, 0.5, 0.5])
    create_key_load()


    nnl,nel=create_model(radiuses,50)
    apply_3d_PBC()
    t2 = time.time()
        # anl=app.feAnalysisMgr
    #     anl.Analyze(1)
    #     tstr='Simcenter Nastran finished'
    #     while True:
    #         if check_last_line_startswith_large(f'E:/key_dof_homo/kdf-{count:04}.log',tstr): break
    #     time.sleep(10)
    #     print((time.time()-t1))
    #     print(f'it = {count}, R = f{radiuses}, es-time-left = {(time.time()-t1)*(len(result)-count)}')
    #     count+=1
    # app.feDeleteAll(True, True, True, True)

