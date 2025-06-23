import pythoncom
import PyFemap
import sys
import win32com.client.gencache
import numpy as np
from femap_integration import create_matirial_iso,create_property
from scipy.spatial import Delaunay, KDTree
from modelling_utils import *


if __name__ == '__main__':
    try:
        win32com.client.gencache.is_readonly=False
        existObj = pythoncom.connect(PyFemap.model.CLSID) #Grabs active model
        app = PyFemap.model(existObj)#Initializes object to active mode
    except:
        sys.exit("femap not open") #Exits program if there is no active femap model

ax_dict={0:[0,1,2],1:[1,0,2],2:[2,1,0]}
eye3=np.eye(3)
steppin=[[-1,-1],[-1,1],[1,1],[1,-1]]
accuracy=0.002
def create_boundary_nodes(nodes:list[float],center,r: float,axis:int,neighbor_nodes)->None:
    global accuracy

    for step in steppin:
        if abs(neighbor_nodes[2*axis+1]-r)<=accuracy: rt=max(r,neighbor_nodes[2*axis+1])
        else: rt=r
        tmp = [center[ax_dict[axis][0]] + 0.5, center[ax_dict[axis][1]] + rt * step[0],
               center[ax_dict[axis][2]] + rt * step[1]]
        nodes.append(tmp[ax_dict[axis][0]])
        nodes.append(tmp[ax_dict[axis][1]])
        nodes.append(tmp[ax_dict[axis][2]])

        if abs(neighbor_nodes[2 * axis] - r) <= accuracy: rt = max(r, neighbor_nodes[2 * axis])
        else:rt = r

        tmp = [center[ax_dict[axis][0]] + 0.5, center[ax_dict[axis][1]] + rt * step[0],
               center[ax_dict[axis][2]] + rt * step[1]]
        tmp[0]=center[ax_dict[axis][0]] - 0.5
        nodes.append(tmp[ax_dict[axis][0]])
        nodes.append(tmp[ax_dict[axis][1]])
        nodes.append(tmp[ax_dict[axis][2]])


def create_intersection_nodes(nodes:list[float],inner_nodes: list[float],center,r1: float,r2: float,axis1:int,axis2:int)->None:
    l1=len(nodes)
    for step in steppin:
        tmp = [center[ax_dict[axis2][0]] + r1, center[ax_dict[axis2][1]] + r2 * step[0],
               center[ax_dict[axis2][2]] + r2 * step[1]]
        nodes.append(tmp[ax_dict[axis2][0]])
        nodes.append(tmp[ax_dict[axis2][1]])
        nodes.append(tmp[ax_dict[axis2][2]])

        tmp[0] = center[ax_dict[axis2][0]] - r1
        nodes.append(tmp[ax_dict[axis2][0]])
        nodes.append(tmp[ax_dict[axis2][1]])
        nodes.append(tmp[ax_dict[axis2][2]])
    for i in range(l1,len(nodes)):
        inner_nodes.append(nodes[i])

def create_slice_nodes(nodes:list[float],inner_nodes: list[float],center,r: float,axis:int,offset:float)->None:
    l1 = len(nodes)

    for step in steppin:
        tmp = [center[ax_dict[axis][0]] + offset, center[ax_dict[axis][1]] + r * step[0],
               center[ax_dict[axis][2]] + r * step[1]]
        nodes.append(tmp[ax_dict[axis][0]])
        nodes.append(tmp[ax_dict[axis][1]])
        nodes.append(tmp[ax_dict[axis][2]])

        tmp[0] = center[ax_dict[axis][0]] - offset
        nodes.append(tmp[ax_dict[axis][0]])
        nodes.append(tmp[ax_dict[axis][1]])
        nodes.append(tmp[ax_dict[axis][2]])

    for i in range(l1,len(nodes)):
        inner_nodes.append(nodes[i])

border=[(-0.5,[0,1,2]),(0.5,[0,1,2]),(-0.5,[1,0,2]),(0.5,[1,0,2]),(-0.5,[2,1,0]),(0.5,[2,1,0])]
def create_contact_nodes(nodes:list[float],center:tuple[float,float,float],rs:list[float],neighbor_rad:list[float]):
    global accuracy

    for i in range(6):
        if rs[border[i][1][0]]<neighbor_rad[i] or neighbor_rad[i]<=0:continue

        if abs(rs[border[i][1][0]]-neighbor_rad[i])<=accuracy:continue

        offset=border[i][0]
        axis=border[i][1][0]
        r=neighbor_rad[i]

        for step in steppin:
            tmp = [center[ax_dict[axis][0]] + offset, center[ax_dict[axis][1]] + r * step[0],
                   center[ax_dict[axis][2]] + r * step[1]]
            nodes.append(tmp[ax_dict[axis][0]])
            nodes.append(tmp[ax_dict[axis][1]])
            nodes.append(tmp[ax_dict[axis][2]])

def create_central_cube(nodes,center,r):
    for i in range(2):
        for j in range(2):
            for k in range(2):
                nodes.append(center[0] + r*(2*i-1))
                nodes.append(center[1] + r*(2*j- 1))
                nodes.append(center[2] + r*(2*k - 1))

def delaunay_triangulation_3d(points):
    """Выполнение триангуляции Делоне в 3D"""
    if len(points) < 4:
        raise ValueError("Need at least 4 points to perform 3D Delaunay triangulation")

    # Выполняем триангуляцию Делоне
    tri = Delaunay(points)

    # Возвращаем симплексы (тетраэдры)
    return tri.simplices


def sort_points_by_angle(points, center,axis):
    center=np.array(center)
    # Удаляем точку center (если она есть)
    mask = ~np.all(points == center, axis=1)
    filtered_points = points[mask]

    # Вычисляем векторы от center к каждой точке
    vectors = filtered_points[:,ax_dict[axis][1:3]] - center[ax_dict[axis][1:3]]

    # Вычисляем углы (в радианах) между осью OX и векторами
    angles = np.arctan2(vectors[:, 1], vectors[:, 0])

    # Сортируем точки по углам
    sorted_indices = np.argsort(angles)
    a=np.arange(len(points))
    sorted_indices=a[mask][sorted_indices]
    return sorted_indices

from icecream import ic
import matplotlib.pyplot as plt


def create_some_elements(nodes,nodeTree,center,r,axis,x1,x2):
    global accuracy


    c1=np.array(center)
    c1[axis]+=x1
    c2 = np.array(center)
    c2[axis] += x2


    plain1=np.array(nodeTree.query_ball_point(c1,r+accuracy,np.inf))
    plain1=plain1[nodes[plain1][:,axis]==c1[axis]]

    if abs(x2)==0.5:acc=accuracy
    else: acc=1e-6
    plain2 = np.array(nodeTree.query_ball_point(c2, r+acc, np.inf))
    plain2 = plain2[nodes[plain2][:,axis] == c2[axis]]


    hex_elem=[]
    wedge_elem=[]
    if len(plain1)==len(plain2)==4:
        create_hex(nodes,np.concatenate([plain1,plain2]),hex_elem)
        return hex_elem,wedge_elem

    if len(plain1)<len(plain2):
        plain1,plain2=plain2,plain1
        c1,c2=c2,c1

    if len(plain2)!=len(plain1):
        plain1_inner=np.array(nodeTree.query_ball_point(c1,r-accuracy,np.inf))
        plain1_inner=plain1_inner[nodes[plain1_inner][:,axis] == c1[axis]]
        plain1_outer=np.setdiff1d(plain1,plain1_inner)
        create_hex(nodes,np.concatenate([plain1_inner,plain2]),hex_elem)

        plain1_inner = plain1_inner[sort_points_by_angle(nodes[plain1_inner], c1, axis)]
        plain1_outer = plain1_outer[sort_points_by_angle(nodes[plain1_outer], c1, axis)]
        plain2 = plain2[sort_points_by_angle(nodes[plain2], c2, axis)]


        for eln in range(4):
            wedge_elem.append(plain1_inner[eln]+1)
            wedge_elem.append(plain1_outer[eln]+1)
            wedge_elem.append(plain2[eln]+1)
            wedge_elem.append(0)

            wedge_elem.append(plain1_inner[eln - 1]+1)
            wedge_elem.append(plain1_outer[eln - 1]+1)
            wedge_elem.append(plain2[eln - 1]+1)
            for _ in range(13): wedge_elem.append(0)

        return hex_elem,wedge_elem

    if len(plain1)==len(plain2)==8:
        plain1_inner = np.array(nodeTree.query_ball_point(c1, r - accuracy, np.inf))
        plain1_inner = plain1_inner[nodes[plain1_inner][:,axis] == c1[axis]]
        plain1_outer = np.setdiff1d(plain1, plain1_inner)

        plain1_inner = plain1_inner[sort_points_by_angle(nodes[plain1_inner], c1,axis)]
        plain1_outer = plain1_outer[sort_points_by_angle(nodes[plain1_outer], c1,axis)]

        plain2_inner = np.array(nodeTree.query_ball_point(c2, r - accuracy, np.inf))
        plain2_inner = plain2_inner[nodes[plain2_inner][:,axis] == c2[axis]]
        plain2_outer = np.setdiff1d(plain2, plain2_inner)

        plain2_inner = plain2_inner[sort_points_by_angle(nodes[plain2_inner], c2,axis)]
        plain2_outer = plain2_outer[sort_points_by_angle(nodes[plain2_outer], c2,axis)]

        hex_elem.append(plain1_inner[0]+1)
        hex_elem.append(plain1_inner[1]+1)
        hex_elem.append(plain1_inner[2]+1)
        hex_elem.append(plain1_inner[3] + 1)

        hex_elem.append(plain2_inner[0] + 1)
        hex_elem.append(plain2_inner[1] + 1)
        hex_elem.append(plain2_inner[2] + 1)
        hex_elem.append(plain2_inner[3] + 1)
        for _ in range(12): hex_elem.append(0)

        for eln in range(4):
            hex_elem.append(plain1_inner[eln]+1)
            hex_elem.append(plain1_outer[eln]+1)
            hex_elem.append(plain1_outer[eln-1]+1)
            hex_elem.append(plain1_inner[eln-1]+1)

            hex_elem.append(plain2_inner[eln]+1)
            hex_elem.append(plain2_outer[eln]+1)
            hex_elem.append(plain2_outer[eln-1]+1)
            hex_elem.append(plain2_inner[eln- 1]+1)
            for _ in range(12): hex_elem.append(0)
    return hex_elem,wedge_elem









def create_out_elements(nodeTree,nodes:list[list[float]],center,rs:list[float],axises:list[int]):
    global accuracy
    if rs[0]==0: return [],[]
    if rs[0]==0.5: return [],[]
    hex_elemnts=[]
    wedge_elements=[]
    if rs[1]==0:
        hex_el,wedge_el=create_some_elements(nodeTree,nodes,center,rs[0],axises[0],-0.5,0.5)
        hex_elemnts+=hex_el
        wedge_elements+=wedge_el
        return hex_elemnts,wedge_elements


    #first strut
    hex_el, wedge_el = create_some_elements(nodeTree, nodes, center, rs[0], axises[0], - 0.5,- rs[0])
    hex_elemnts += hex_el
    wedge_elements+=wedge_el

    hex_el, wedge_el = create_some_elements(nodeTree, nodes, center, rs[0], axises[0], 0.5,  rs[0])
    hex_elemnts += hex_el
    wedge_elements += wedge_el

    #second strut
    hex_el, wedge_el = create_some_elements(nodeTree, nodes, center, rs[1], axises[1], - 0.5,- rs[0])
    hex_elemnts += hex_el
    wedge_elements += wedge_el

    hex_el, wedge_el = create_some_elements(nodeTree, nodes, center, rs[1], axises[1],0.5, rs[0])
    hex_elemnts += hex_el
    wedge_elements += wedge_el

    if rs[2]==0:
        return hex_elemnts,wedge_elements

    #third strut
    hex_el, wedge_el = create_some_elements(nodeTree, nodes, center, rs[2], axises[2], - 0.5,- rs[0])
    hex_elemnts += hex_el
    wedge_elements += wedge_el

    hex_el, wedge_el = create_some_elements(nodeTree, nodes, center, rs[2], axises[2], 0.5,rs[0])
    hex_elemnts += hex_el
    wedge_elements += wedge_el

    return hex_elemnts, wedge_elements


def are_points_coplanar_fast(arr, tol=1e-6):

    a,b,c,d=arr

    ab = b - a
    ac = c - a
    ad = d - a

    # Векторное произведение ab × ac
    cross = np.cross(ab, ac)

    # Скалярное произведение (ab × ac) · ad
    volume = np.dot(cross, ad)

    return abs(volume) < tol


import numpy as np


def calculate_wedge_volume(vertices):
    """
    Вычисляет объём wedge-элемента (6 вершин, 5 граней) путём разбиения на 3 тетраэдра.

    Параметры:
        vertices: список из 6 точек в формате [(x1, y1, z1), (x2, y2, z2), ..., (x6, y6, z6)].
                 Порядок вершин должен соответствовать структуре wedge:
                 - Основание: вершины 0, 1, 2 (нижний треугольник)
                 - Верх: вершины 3, 4, 5 (верхний треугольник)

    Возвращает:
        Объём wedge-элемента.
    """
    if len(vertices) != 6:
        raise ValueError("Wedge-элемент должен иметь 6 вершин.")

    # Разбиваем wedge на 3 тетраэдра:
    # 1. Тетраэдр 1: 0, 1, 2, 3
    # 2. Тетраэдр 2: 1, 2, 3, 4
    # 3. Тетраэдр 3: 2, 3, 4, 5
    tetrahedra = [
        [vertices[0], vertices[1], vertices[2], vertices[3]],
        [vertices[1], vertices[2], vertices[3], vertices[4]],
        [vertices[2], vertices[3], vertices[4], vertices[5]]
    ]

    total_volume = 0.0

    for tetra in tetrahedra:
        A, B, C, D = np.array(tetra[0]), np.array(tetra[1]), np.array(tetra[2]), np.array(tetra[3])

        AB = B - A
        AC = C - A
        AD = D - A

        cross = np.cross(AC, AD)
        dot = np.dot(AB, cross)

        volume = abs(dot) / 6.0
        total_volume += volume

    return total_volume


def create_hex(nodes,node_id,elements):
    if len(set(node_id))!=8:
        return 0

    node_id=node_id[nodes[node_id,2].argsort()]
    c=nodes[node_id].sum(0)/8
    node_id[0:4]=node_id[sort_points_by_angle(nodes[node_id[0:4]],c,2)]
    node_id[4::] = node_id[4+sort_points_by_angle(nodes[node_id[4::]], c, 2)]
    for var in node_id:
            elements.append(var+1)
    for _ in range(12): elements.append(0)

def create_wedge(nodes,node_id,elements,main_ax):
    if len(set(node_id)) != 6: return 0
    if calculate_wedge_volume(nodes[node_id])<1e-6:return 0

    node_id=node_id[nodes[node_id,main_ax].argsort()]
    c=nodes[node_id].sum(0)/6
    node_id[0:3]=node_id[sort_points_by_angle(nodes[node_id[0:3]],c,main_ax)]
    node_id[3::] = node_id[3+sort_points_by_angle(nodes[node_id[3::]], c, main_ax)]
    count=0
    for var in node_id:
        if count==3:elements.append(0)
        elements.append(var+1)
        count+=1
    for _ in range(13): elements.append(0)

def create_wedges(nodes,center,nodeTree,rs,axises,c,el_nodes,i,a1,a2,wedge_elememts):
    wax = ax_dict[axises[i]][a1]
    waxs = ax_dict[axises[i]][a2]
    w_nodes = el_nodes[nodes[el_nodes][:, wax] > center[wax]].flatten()
    c1 = c.copy()
    c1[wax] += rs[0]
    c1[waxs] += rs[0]

    c2 = c.copy()
    c2[wax] += rs[0]
    c2[waxs] -= rs[0]

    c1 = nodeTree.query(c1)[1]
    c2 = nodeTree.query(c2)[1]
    create_wedge(nodes, np.concatenate([w_nodes, [c1], [c2]]), wedge_elememts, waxs)

    w_nodes = el_nodes[nodes[el_nodes][:, wax] < center[wax]].flatten()
    c1 = c.copy()
    c1[wax] -= rs[0]
    c1[waxs] += rs[0]

    c2 = c.copy()
    c2[wax] -= rs[0]
    c2[waxs] -= rs[0]

    c1 = nodeTree.query(c1)[1]
    c2 = nodeTree.query(c2)[1]
    create_wedge(nodes, np.concatenate([w_nodes, [c1], [c2]]), wedge_elememts, waxs)
def create_inner_elements(nodes,nodeTree,center,rs,axises,inr):
    hex_elements=[]
    wedge_elememts=[]
    if rs[1]==rs[2]==0:return hex_elements,wedge_elememts

    inner_cube=np.array(nodeTree.query_ball_point(center,inr+1e-6,np.inf))
    create_hex(nodes,inner_cube,hex_elements)


    for i in range(3):
        if rs[i]==0:r=rs[0]
        else: r=rs[i]

        c=np.array(center)
        c[axises[i]]+=rs[0]
        out_side=np.array(nodeTree.query_ball_point(c,r+1e-6,np.inf))
        if len(out_side)!=0:
            out_side=out_side[nodes[out_side][:,axises[i]]==c[axises[i]]].flatten()
            el_nodes=np.concatenate([out_side,inner_cube[nodes[inner_cube][:,axises[i]]==(center[axises[i]]+inr)]])
            create_hex(nodes,el_nodes,hex_elements)

            if i!=0:
                create_wedges(nodes,center,nodeTree,rs,axises,c,el_nodes,i,1,2,wedge_elememts)
                create_wedges(nodes, center, nodeTree, rs, axises, c, el_nodes, i, 2, 1,wedge_elememts)

        if len(out_side) != 0:
            c = np.array(center)
            c[axises[i]] -= rs[0]
            out_side = np.array(nodeTree.query_ball_point(c, r+1e-6, np.inf))
            out_side = out_side[nodes[out_side][:,axises[i]] == c[axises[i]]].flatten()
            el_nodes=np.concatenate([out_side,inner_cube[nodes[inner_cube][:,axises[i]]==(center[axises[i]]-inr)]])
            create_hex(nodes,el_nodes,hex_elements)

            if i!=0:
                create_wedges(nodes,center,nodeTree,rs,axises,c,el_nodes,i,1,2,wedge_elememts)
                create_wedges(nodes, center, nodeTree, rs, axises, c, el_nodes, i, 2, 1,wedge_elememts)
    return hex_elements,wedge_elememts

def create_inner_elements_cc(nodes,nodeTree,center,rs,axises,inr,neighbor_radii):
    if rs[0]==0.5:inr=neighbor_radii[neighbor_radii!=0].min()
    hex_elements=[]
    wedge_elememts=[]
    if rs[1]==rs[2]==0 and rs[0] != 0.5 :return hex_elements,wedge_elememts

    inner_cube=np.array(nodeTree.query_ball_point(center,inr+1e-6,np.inf))
    create_hex(nodes,inner_cube,hex_elements)


    for i in range(3):
        if rs[i]==0:r=rs[0]
        else: r=rs[i]
        if rs[0]==0.5:
            r=neighbor_radii[2*axises[i]+1]
            if r == 0: r = 0.5
        c=np.array(center)
        c[axises[i]]+=rs[0]
        out_side=np.array(nodeTree.query_ball_point(c,r+1e-6,np.inf))
        if len(out_side) != 0:
            out_side=out_side[nodes[out_side][:,axises[i]]==c[axises[i]]].flatten()
            el_nodes=np.concatenate([out_side,inner_cube[nodes[inner_cube][:,axises[i]]==(center[axises[i]]+inr)]])
            create_hex(nodes,el_nodes,hex_elements)

            create_wedges(nodes,center,nodeTree,rs,axises,c,el_nodes,i,1,2,wedge_elememts)
            create_wedges(nodes, center, nodeTree, rs, axises, c, el_nodes, i, 2, 1,wedge_elememts)


        if rs[0] == 0.5:
            r = neighbor_radii[2 * axises[i]]
            if r==0: r=0.5
        c = np.array(center)
        c[axises[i]] -= rs[0]
        out_side = np.array(nodeTree.query_ball_point(c, r+1e-6, np.inf))
        if len(out_side) != 0:
            out_side = out_side[nodes[out_side][:,axises[i]] == c[axises[i]]].flatten()
            el_nodes=np.concatenate([out_side,inner_cube[nodes[inner_cube][:,axises[i]]==(center[axises[i]]-inr)]])
            create_hex(nodes,el_nodes,hex_elements)


            create_wedges(nodes,center,nodeTree,rs,axises,c,el_nodes,i,1,2,wedge_elememts)
            create_wedges(nodes, center, nodeTree, rs, axises, c, el_nodes, i, 2, 1,wedge_elememts)
    return hex_elements,wedge_elememts

def create_unit_cell_mesh(radii:list,el_id: int,center: tuple[float,float,float],neighbor_radii: list[float]):
    global max_elm_id
    global max_node_id
    global accuracy
    r=radii[3*el_id:3*el_id+3]
    if np.array_equal(r, np.zeros(3)): return 1

    ic(r)
    ic(neighbor_radii)
    neighbor_radii[(neighbor_radii+accuracy)>0.5]=0.5


    nodes = []
    inner_nodes = []



    ri=np.argsort(-r)
    axises=np.array([0,1,2])[ri]
    r=r[ri]

    if r[0]+accuracy>0.5:
        r=np.array([0.5,0.5,0.5])
        create_central_cube(nodes, center, neighbor_radii[neighbor_radii!=0].min())
    if r[1]+accuracy >= r[0]: r[1]=r[0]
    if r[2]+accuracy >= r[1]: r[2] = r[1]

    back = [np.where(ri == 0)[0], np.where(ri == 1)[0], np.where(ri == 2)[0]]
    create_contact_nodes(nodes, center, r[back].flatten(), neighbor_radii)

    elements=[]
    in_cube_r=0
    if r[0]>0:
        create_boundary_nodes(nodes,center,r[0],axises[0],neighbor_radii)
        if r[1]>0:
            create_boundary_nodes(nodes,center, r[1], axises[1],neighbor_radii)
            create_intersection_nodes(nodes,inner_nodes,center,r[0],r[1],axises[0],axises[1])

            create_slice_nodes(nodes, inner_nodes, center, r[0], axises[0], r[0])
            create_slice_nodes(nodes, inner_nodes, center, r[0], axises[0], -r[0])

            if r[2]>0:
                create_central_cube(nodes, center, r[2])
                in_cube_r=r[2]
                create_boundary_nodes(nodes, center, r[2], axises[2],neighbor_radii)
                create_intersection_nodes(nodes,inner_nodes, center, r[0], r[2], axises[0], axises[2])
            else:
                create_central_cube(nodes,center,r[1])
                in_cube_r=r[1]

    tnodes = []
    for i in range(len(nodes) // 3): tnodes.append((nodes[3 * i], nodes[3 * i + 1], nodes[3 * i + 2]))
    nodes = list(set(tnodes))
    nodes = np.array(nodes)  # .reshape(len(nodes)//3,3)
    inner_nodes = np.array(inner_nodes).reshape(len(inner_nodes) // 3, 3)
    node = app.feNode
    print('Creating Nodes, return code = ',
          node.PutCoordArray(len(nodes.flatten()) // 3, np.arange(1, len(nodes.flatten()) // 3 + 1)+max_node_id, nodes.flatten()))


    node_id = KDTree(nodes)
    hex_elements=[]
    wedge_elements=[]
    hex_elements,wedge_elements=create_inner_elements(nodes,node_id,center,r,axises,in_cube_r)
    hex_el,wedge_el=create_out_elements(nodes,node_id,center,r,axises)
    hex_elements=hex_elements+hex_el
    wedge_elements = wedge_elements + wedge_el

    # for pn in range(1,len(nodes)):
    #     if pn not in hex_elements and pn not in wedge_elements:
    #         ic(center)
    if r[0]==0.5:hex_elements,wedge_elements=create_inner_elements_cc(nodes,node_id,center,r,axises,in_cube_r,neighbor_radii)

    elem = app.feElem

    hex_elements = np.array(hex_elements)
    hex_elements[hex_elements != 0] += max_node_id
    wedge_elements = np.array(wedge_elements)
    wedge_elements[wedge_elements != 0] += max_node_id

    nelh = len(hex_elements) // 20
    print('hex_elem',elem.PutAllArray(nelh, np.arange(1, nelh + 1)+max_elm_id, [1] * nelh, [25] * nelh, [8] * nelh, [1] * nelh, [124] * nelh,
                           [0] * 2 * nelh, [0] * 3 * nelh, [0] * nelh * 6, [0] * nelh * 12, [0] * nelh, [0] * nelh,
                           hex_elements,
                           [0] * 2 * nelh, [0] * 2 * nelh))

    nelw = len(wedge_elements) // 20
    print('wedge',elem.PutAllArray(nelw, np.arange(nelh + 1, nelh+nelw + 1)+max_elm_id, [1] * nelw, [25] * nelw, [7] * nelw, [1] * nelw, [124] * nelw,
                           [0] * 2 * nelw, [0] * 3 * nelw, [0] * nelw * 6, [0] * nelw * 12, [0] * nelw, [0] * nelw,
                           wedge_elements,
                           [0] * 2 * nelw, [0] * 2 * nelw))
    max_node_id+=len(nodes)
    max_elm_id+=(nelh+nelw)

def prepare_radii(radii):
    for i in range(len(radii)//3):
        r=radii[3*i:3*i+3]
        ri = np.argsort(-r)
        axises = np.array([0, 1, 2])[ri]
        r = r[ri]
        if r[0] + accuracy > 0.5:
            r = np.array([0.5, 0.5, 0.5])
        if r[1] + accuracy >= r[0]: r[1] = r[0]
        if r[2] + accuracy >= r[1]: r[2] = r[1]
        for j,ax in enumerate(axises):
            radii[3*i+ax]=r[j]
def create_mesh(r,nelx,nely,nelz):
    prepare_radii(r)
    for x in range(nelx):
        for y in range(nely):
            for z in range(nelz):
                center=(0.5+x,0.5+y,0.5+z)
                el_id=x+nelx*y+nelx*nely*z
                neighbor_radii=[]
                if x==0: neighbor_radii.append(0)
                else: neighbor_radii.append(r[(x-1+nelx*y+nelx*nely*z)*3])
                if x==nelx-1: neighbor_radii.append(0)
                else: neighbor_radii.append(r[(x+1+nelx*y+nelx*nely*z)*3])

                if y==0: neighbor_radii.append(0)
                else: neighbor_radii.append(r[(x+nelx*(y-1)+nelx*nely*z)*3+1])
                if y==nely-1: neighbor_radii.append(0)
                else: neighbor_radii.append(r[(x+nelx*(y+1)+nelx*nely*z)*3+1])

                if z==0: neighbor_radii.append(0)
                else: neighbor_radii.append(r[(x+nelx*y+nelx*nely*(z-1))*3+2])
                if z==nelz-1: neighbor_radii.append(0)
                else: neighbor_radii.append(r[(x+nelx*y+nelx*nely*(z+1))*3+2])

                create_unit_cell_mesh(r,el_id,center,np.array(neighbor_radii))


if __name__ == '__main__':

    from modelling_utils import profile_0012
    max_node_id=0
    max_elm_id=0
    nelx,nely,nelz=210,10,55
    rmin = 2
    create_matirial_iso(200, 0.3)
    create_property()

    from model import model
    from scipy.sparse import coo_matrix
    elasticity_model = model('weights', 'scalers')
    x = np.zeros(3 * nely * nelx * nelz, dtype=float)
    x1 = np.load('optimization_results/opt_res_2025-06-05_19-11-05/radii_50.npy').flatten()

    xmask = np.zeros(nelx * nely * nelz, dtype=bool)
    for xx in range(nelx):
        for yy in range(nely):
            for zz in range(nelz):
                center = (xx + 0.5, yy + 0.5,zz+0.5)
                idx = xx + nelx * yy + nelx * nely * zz
                xmask[idx] = profile_0012(center, nelx, nely,55,35)
    x[np.kron(xmask, np.ones(3)).flatten().astype(bool)] = x1



    outer_nodes_arr = outer_nodes(nelx, nely, nelz, xmask)
    f = apply_presure_load_on_prof0012(outer_nodes_arr, nelx, nely, nelz, 55, 35).reshape(3*(nelx+1)*(nely+1)*(nelz+1), 1)

    fmask = np.zeros(3 * nelx * nely * nelz, dtype=bool)
    for xx in range(nelx):
        for yy in range(nely):
            for zz in range(nelz):
                idx = xx + nelx * yy + nelx * nely * zz
                nodes_arr = np.array([xx + (nelx + 1) * yy + (nelx + 1) * (nely + 1) * zz,
                                      xx + 1 + (nelx + 1) * yy + (nelx + 1) * (nely + 1) * zz,
                                      xx + (nelx + 1) * (yy + 1) + (nelx + 1) * (nely + 1) * zz,
                                      xx + (nelx + 1) * yy + (nelx + 1) * (nely + 1) * (zz + 1),
                                      xx + 1 + (nelx + 1) * (yy + 1) + (nelx + 1) * (nely + 1) * zz,
                                      xx + (nelx + 1) * (yy + 1) + (nelx + 1) * (nely + 1) * (zz + 1),
                                      xx + 1 + (nelx + 1) * yy + (nelx + 1) * (nely + 1) * (zz + 1),
                                      xx + 1 + (nelx + 1) * (yy + 1) + (nelx + 1) * (nely + 1) * (zz + 1)],
                                     dtype=np.int32)
                if np.abs(f[3 * nodes_arr].sum()) + np.abs(f[3 * nodes_arr + 1].sum()) + np.abs(
                    f[3 * nodes_arr + 2].sum()) != 0: fmask[3 * idx:3 * idx + 3] = True
    fmask = fmask[np.kron(xmask, np.ones(3)).flatten().astype(bool)].astype(bool)
    # x = x.flatten()
    # x = np.ones(3 * nely * nelx * nelz, dtype=float) * 0.2
    # Thn=np.load('ahaha.npy')
    # x=x*np.kron(Thn,np.ones(3)).flatten()

    nfilter = int(nelx * nely * nelz * ((2 * (np.ceil(rmin) - 1) + 1) ** 3))
    iH = np.zeros(nfilter)
    jH = np.zeros(nfilter)
    sH = np.zeros(nfilter)
    cc = 0
    for z in range(nelz):
        for i in range(nelx):
            for j in range(nely):
                row = i * nely + j + z * (nelx * nely)
                kk1 = int(np.maximum(i - (np.ceil(rmin) - 1), 0))
                kk2 = int(np.minimum(i + np.ceil(rmin), nelx))
                ll1 = int(np.maximum(j - (np.ceil(rmin) - 1), 0))
                ll2 = int(np.minimum(j + np.ceil(rmin), nely))
                mm1 = int(np.maximum(z - (np.ceil(rmin) - 1), 0))
                mm2 = int(np.minimum(z + np.ceil(rmin), nelz))
                for mm in range(mm1, mm2):
                    for kk in range(kk1, kk2):
                        for ll in range(ll1, ll2):
                            col = kk * nely + ll + mm * (nelx * nely)
                            fac = rmin - np.sqrt((i - kk) * (i - kk) + (j - ll) * (j - ll) + (z - mm) * (z - mm))
                            iH[cc] = row
                            jH[cc] = col
                            sH[cc] = np.maximum(0.0, fac)
                            cc = cc + 1
    # Finalize assembly and convert to csc format
    H = coo_matrix((sH, (iH, jH)), shape=(nelx * nely * nelz, nelx * nely * nelz)).tocsc()
    Hs = H.sum(1)

    xPhys = x.copy()
    """applying filter"""
    for i in range(3):
        xPhys[i::3] = np.asarray(H * x[i::3][np.newaxis].T / Hs)[:, 0]


    rho = elasticity_model.predict_value_density(xPhys[0::3], xPhys[1::3], xPhys[2::3])



    a = 1e4
    b = 0.1
    Th = np.array(list(map(lambda zu: 1 / 2 * (1 + np.tanh(a * (zu - b))), rho))).flatten()
    xPhys = xPhys * np.kron(Th, np.ones(3)).flatten()
    xPhys[np.logical_and(np.kron(np.array(Th), np.ones(3)).flatten() > 0.1, xPhys < 0.1)] = 0.1
    print((rho*Th).sum()/nelx/nely/nelz)

    Th[fmask[0::3]]=1
    thresh = 0.07
    xPhys[xPhys < thresh] = 0
    xPhys[np.logical_and(xPhys > thresh, xPhys < 0.1)] = 0.1
    xPhys[np.logical_not(np.kron(xmask,np.ones(3)).flatten().astype(bool))]=0
    xPhys[xPhys[fmask]<0.1]=0.1
    create_mesh(xPhys, nelx, nely, nelz)
    #create_unit_cell_mesh(np.array([0.        , 0.        , 0.11028592]),0,(0.5,0.5,0.5),np.array([0.       , 0.       , 0.       , 0.       , 0.       , 0.1131877]))