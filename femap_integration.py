import time

import pythoncom
import PyFemap
import sys
import win32com.client.gencache
import numpy as np

from PBC_femap import apply_2d_PBC,apply_3d_PBC
#from model import model

try:
    win32com.client.gencache.is_readonly=False
    existObj = pythoncom.connect(PyFemap.model.CLSID) #Grabs active model
    app = PyFemap.model(existObj)#Initializes object to active mode
except:
    sys.exit("femap not open") #Exits program if there is no active femap model

def create_matirial_iso(E,nu):
    mt = app.feMatl
    mt.title = 'mymat'
    mt.Put(1)
    mt.PutValueArray(2,[1,1],[0,6],[E,nu])
    mt.AutoComplete()

def create_matirial_aniso(Ce):
    Ce=Ce.flatten()
    mt = app.feMatl
    mt.title = 'mymat'
    mt.Put(1)
    mt.PutValueArray(21, np.ones(21), np.arange(9,30,dtype=np.int32), [Ce[0],Ce[1],Ce[2],0,0,0,
                                                               Ce[3],Ce[4],0,0,0,
                                                               Ce[5],0,0,0,
                                                               Ce[6],0,0,
                                                               Ce[7],0,
                                                               Ce[8]])
    mt.type = 4
    mt.Put(1)

def create_property():
    pr=app.feProp
    pr.type=25
    pr.matlID=1
    pr.Put(1)

def create_box(length):
    ids=[]
    xyz=[]
    count=1
    for k in range(length+1):
        for j in range(length + 1):
            for i in range(length + 1):
                ids.append(count)

                xyz.append(i)
                xyz.append(j)
                xyz.append(k)
                count+=1
    node=app.feNode
    node.PutCoordArray(len(ids),ids,xyz)


    entID=0
    nodes=[]
    count=1
    for i in range(length):
        for j in range(length):
            for k in range(length):
                nodes.append(i + (j+1) * (length + 1) + k * (length + 1) * (length + 1) + 1)
                nodes.append(i + (j + 1) * (length + 1) + (k + 1)* (length + 1) * (length + 1) + 1)
                nodes.append(i + j * (length + 1) + (k + 1) * (length + 1) * (length + 1) + 1)
                nodes.append(i + j * (length + 1) + k * (length + 1) * (length + 1) + 1)

                nodes.append(i+1 + (j + 1) * (length + 1) + k * (length + 1) * (length + 1) + 1)
                nodes.append(i+1 + (j + 1) * (length + 1) + (k + 1) * (length + 1) * (length + 1) + 1)
                nodes.append(i+1 + j * (length + 1) + (k + 1) * (length + 1) * (length + 1) + 1)
                nodes.append(i+1 + j * (length + 1) + k * (length + 1) * (length + 1) + 1)








                for _ in range(12): nodes.append(0)
                count+=1
    elem=app.feElem
    nel=count-1
    print(elem.PutAllArray(nel,np.arange(1,nel+1),[1]*nel,[25]*nel,[8]*nel,[1]*nel,[124]*nel,[0]*2*nel,[0]*3*nel,[0]*nel*6,[0]*nel*12,[0]*nel,[0]*nel,nodes,[0]*2*nel,[0]*2*nel))
    node.PutCoordArray(1, len(ids)+1, [length/2,length/2,length+0.1])

def create_nodal_bc(length):
    bcset=app.feBCSet
    bcset.title='first'
    setID = 1
    bcset.Put(setID)


    bcd=app.feBCDefinition
    bcd.SetID=setID
    bcd.OnType  = 7
    bcd.DataType = 18
    bcd.title  = "MyDef"
    bcd.Put(1)

    myset=app.feSet
    myset.AddRange(1,(length+1)**2,1)
    length=0
    bc=app.feBCNode
    bc.color = 115
    bc.layer = 1
    bc.expanded = False
    bc.SetID = 1
    bc.BCDefinitionID = 1
    bc.NonZeroConstraint = False
    bc.vdof = (True, True, True, True, True, True)
    bc.Add(myset.ID, True, True, True, True, True, True)


if __name__ == '__main__':
    ls = app.feLoadSet
    ld = app.feLoadDefinition
    le = app.feLoadMesh


    ls.title = f'F{1}'
    setID = 1
    ls.Put(setID)

    ld.SetID = setID
    # if i<3:ld.LoadType = PyFemap.constants.FLT_NFORCE
    # else:ld.LoadType = PyFemap.constants.FLT_NMOMENT
    ld.DataType = PyFemap.constants.FT_NODE
    ld.title = f'F{1}'
    ld.Put(1)

    le.meshID = 1
    le.vdof = (True,True,True)
    le.layer = 1
    le.vload = [1]
    le.SetID = setID
    le.LoadDefinitionID = 1

    t1=time.time()
    folder = 'optimization_results/opt_res_2025-06-06_17-34-33'
    nelx,nely,nelz=400,16,400
    xmask=np.load(f'{folder}/domain.npy')
    f=np.load(f'{folder}/force.npy')

    node_id = 1
    x,y,z=0,0,0
    h_voxel=1
    nodes = {}
    for vx in range(nelx):
        for vy in range(nely):
            for vz in range(nelz):
                idx = vx + nelx * vy + nelx * nely * vz
                if not xmask[idx]: continue

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

    for xx in range(nelx+1):
        for yy in range(nely+1):
            for zz in range(nelz+1):
                idx=xx+(nelx+1)*yy+(nelx+1)*(nely+1)*zz
                if f[3*idx:3*idx+3].sum()==0:continue
                nid = nodes.get((xx,yy,zz))
                if nid:
                    myset = app.feSet
                    myset.Add(nid)
                    print(f[3*idx])
                    le.Add(myset.ID, PyFemap.constants.FLT_NFORCE, 0, (True, True, True), [float(f[3*idx]),float(f[3*idx+1]),float(f[3*idx+2]), 0, 0],
                           [0, 0, 0, 0, 0])
                else:
                    print(1)
                    continue





    print((time.time()-t1))



