import os
import copy
import numpy as np
np.set_printoptions(precision=2, threshold=20, suppress=True)

import pyNastran
pkg_path = pyNastran.__path__[0]

from pyNastran.utils import print_bad_path
from pyNastran.op2.op2 import read_op2
from pyNastran.utils import object_methods, object_attributes
from pyNastran.utils.nastran_utils import run_nastran
import matplotlib.pyplot  as plt

op2_filename = f'D:/tests/Новая папка/m-0001.op2'
op2 = read_op2(op2_filename, build_dataframe=True, debug=False)
disp=op2.displacements[1].data

nelx,nely,nelz=400,16,400
folder = 'optimization_results/opt_res_2025-06-06_17-34-33'
xmask=np.load(f'{folder}/domain.npy')
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

fid,ax=plt.subplots()
u=[]
for zz in range(nelz+1):
    u_slice=[]
    for xx in range(nelx+1):
        for yy in range(nely+1):
            nid=nodes.get((xx,yy,zz))
            if nid:
                u_slice.append(disp[0,nid-1,1])
    u_slice=np.array(u_slice)
    u.append(u_slice.min())

u=np.array(u)
norm=np.abs(u).max()
ax.plot(np.arange(len(u)),-u/norm,label="Максимальное перемещение в сечении")


u=[]
for zz in range(nelz+1):
    u_slice=[]
    for xx in range(nelx+1):
        for yy in range(nely+1):
            nid=nodes.get((xx,yy,zz))
            if nid:
                u_slice.append(disp[0,nid-1,1])
    u_slice=np.array(u_slice)
    u.append(u_slice.max())

u=np.array(u)

ax.plot(np.arange(len(u)),-u/norm,label="Минимальное перемещение в сечении")
plt.grid(True)
plt.legend()

plt.show()
