# import time
# from scipy.sparse import coo_matrix
# from scipy.sparse.linalg import cg
# import pyamg
# import numpy as np

# from main import plot_disp

# from visual_utils import dump, dump_new2,dump_new
# from scipy.linalg import solve
# from problem import Problem
# from modelling_utils import *
from model import model
from visual_utils import write_data_on_hexmesh
import numpy as np
from scipy.sparse import csc_matrix,coo_matrix
from scipy.sparse import save_npz, load_npz
import pythoncom
import PyFemap
import sys
import win32com.client.gencache
import matplotlib.pyplot as plt


try:
    win32com.client.gencache.is_readonly=False
    existObj = pythoncom.connect(PyFemap.model.CLSID) #Grabs active model
    app = PyFemap.model(existObj)#Initializes object to active mode
except:
    sys.exit("femap not open") #Exits program if there is no active femap model

nelx,nely,nelz=400,16,400
volfrac=0.1
rstart=0.2
rmin=1.5
target_volf=0.1
commentary='1'
folder='optimization_results/opt_res_2025-06-06_17-34-33'
ndof = 3 * (nelx + 1) * (nely + 1) * (nelz + 1)
"""setting initial design and optimization domain"""
specific_domain = False
# x=np.random.beta(8,2,size=3*nely * nelx * nelz)*0.49
x = np.ones(3 * nely * nelx * nelz, dtype=float) * rstart
x1 = np.load(f'{folder}/radii_1000.npy')
n=len(x1)
xmask=np.load(f'{folder}/domain.npy')

f=np.load(f'{folder}/force.npy').flatten()
"""setting mask for loaded units"""
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
                                  xx + 1 + (nelx + 1) * (yy + 1) + (nelx + 1) * (nely + 1) * (zz + 1)], dtype=np.int32)
            if np.abs(f[3 * nodes_arr].sum()) + np.abs(f[3 * nodes_arr + 1].sum()) + np.abs(
                f[3 * nodes_arr + 2].sum()) != 0: fmask[3 * idx:3 * idx + 3] = True


fmask = fmask[np.kron(xmask, np.ones(3)).flatten().astype(bool)]




elasticity_model = model('weights','scalers')
Ce0=elasticity_model.predict_value_CE([0.1],[0.1],[0.1])[0]*0.0001
print(Ce0)

H=np.load('wing_H.npy',allow_pickle=True).item()
Hs=H.sum(1)
xPhys=x1.copy()
for i in range(3):
    xPhys[i::3] = np.asarray(H * x1[i::3][np.newaxis].T / Hs)[:, 0]

rho = elasticity_model.predict_value_density(xPhys[0::3], xPhys[1::3], xPhys[2::3]).flatten()
a = 1e5
b = 0.1
Th = np.array(list(map(lambda x: 1 / 2 * (1 + np.tanh(a * (x - b))), rho))).flatten()
Th[fmask[0::3]]=1

minimalr=0.1
xPhys[np.logical_and(np.kron(Th, np.ones(3)).flatten() > 0.1, xPhys < minimalr)] = minimalr
if fmask is not None:
    xPhys[fmask]=np.clip(xPhys[fmask],a_min=0.1,a_max=None)


rho = elasticity_model.predict_value_density(xPhys[0::3], xPhys[1::3],
                                                       xPhys[2::3]).flatten()


rho = rho * Th
x=np.zeros(3*nelx*nely*nelz)
x[np.kron(xmask,np.ones(3)).flatten().astype(bool)]=xPhys
Th_global=np.zeros(nelx*nely*nelz)-1
Th_global[xmask]=Th
rho_global=np.zeros(nelx*nely*nelz)
rho_global[xmask]=rho



# disp1 = np.load('u_test_bending.npy')
# xmask=np.load('optimization_results/opt_res_2025-06-06_17-34-33/domain.npy')
# disp=np.zeros(3*(nelx+1)*(nely+1)*(nelz+1))
# print(xmask.sum())

# ndof=3*(nelx+1)*(nely+1)*(nelz+1)
# dof_mask=np.zeros(ndof,dtype=bool)

# dof_mask=np.load('wing_dof_mask.npy')
# # np.save('wing_dof_mask.npy',dof_mask)
# disp[dof_mask] = disp1.flatten()

# u=[0]
# for zz in range(400):
#     id1=(nelx+1)*(nely+1)*zz
#     id2=nelx+(nelx+1)*nely+(nelx+1)*(nely+1)*zz
#     u_slice=disp[3*id1+1:(3*id2+3):3]
#     u_slice=u_slice[u_slice!=0]


#     if len(u_slice)>0:
#         umin=u_slice.max()
#         if abs(abs(umin)-abs(u[-1]))>10:u.append(u[-1])
#         else:u.append(umin)
#     else:
#         u.append(u[-1])
#         print(zz)
#     # if u[-1]<-100:u[-1]=u[-2]
#     # if u[-1] > 1000 : u[-1] = u[-2]

# fig, ax = plt.subplots()
# ax.plot(np.arange(len(u)),u)
# ax.set_xlim([0,nelz])
# ax.set_ylim([-200,200])
# print(len(u))

# plt.show()

# elm_disp=np.zeros(nelx*nely*nelz)
# for xx in range(nelx):
#     for yy in range(nely):
#         for zz in range(nelz):
#             idx=xx+nelx*yy+nelx*nely*zz
#             nodes=np.array([xx+(nelx+1)*yy+(nelx+1)*(nely+1)*zz,xx+1+(nelx+1)*yy+(nelx+1)*(nely+1)*zz,
#                    xx+(nelx+1)*(yy+1)+(nelx+1)*(nely+1)*zz,xx+(nelx+1)*yy+(nelx+1)*(nely+1)*(zz+1),
#                    xx+1+(nelx+1)*(yy+1)+(nelx+1)*(nely+1)*zz,xx+1+(nelx+1)*yy+(nelx+1)*(nely+1)*(zz+1),
#                    xx+(nelx+1)*(yy+1)+(nelx+1)*(nely+1)*(zz+1),xx+1+(nelx+1)*(yy+1)+(nelx+1)*(nely+1)*(zz+1)])
#             eld=disp[3*nodes+1]
#             elm_disp[idx]=eld.mean()
# print(elm_disp[xmask])
# write_data_on_hexmesh('wing_disps.vtu',nelx+1,nely+1,nelz+1,Th=Th_global,eld=elm_disp)



C = elasticity_model.predict_value_CE(xPhys[0::3], xPhys[1::3], xPhys[2::3])



Ce = np.zeros((n // 3, 21))
idx=[0,1,2,6,7,11,15,18,20]
for i in range(9):Ce[:,idx[i]]=C[:,i]
Ce_global=np.zeros((nelx*nely*nelz,21))
Ce_global[xmask,:]=Ce[:,:]



mt = app.feMatl
mt.type = 4
eln=(Th>0.1).sum()
print(eln)
for i in range(eln):
    mt.Put(i + 1)
ret=mt.PutValueArray(21*eln, np.kron(np.arange(1,eln+1),np.ones(21)).flatten(),
                     np.kron(np.ones(eln),np.arange(9, 30, dtype=np.int32)).flatten(), Ce_global[Th_global>0.1,:].flatten())
pr = app.feProp
pr.type = 25
for i in range(eln):
    pr.matlID = i+1
    pr.Put(i+1)



nodes={}
elements=[]
node_id=1
h_voxel=1
x,y,z=0,0,0
prop_id=[]
porp_id_count=1
end_prop=1000000
for vx in range(nelx):
    for vy in range(nely):
        for vz in range(nelz):
            idx=vx+nelx*vy+nelx*nely*vz
            if Th_global[idx] < 0: continue
            if Th_global[idx]>0.1:
                prop_id.append(porp_id_count)
                porp_id_count+=1
            else: prop_id.append(end_prop)
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

xyz=[]
for key in nodes:
    xyz.append(key[0])
    xyz.append(key[1])
    xyz.append(key[2])
ids=np.arange(1,node_id)
node=app.feNode
node.PutCoordArray(len(ids),ids,xyz)

mt.Put(end_prop)
eln=1

mt.PutValueArray(21*eln, [end_prop]*21,
                     np.kron(np.ones(eln),np.arange(9, 30, dtype=np.int32)).flatten(), [Ce0[0],Ce0[1],Ce0[2],0,0,0,
                                                               Ce0[3],Ce0[4],0,0,0,
                                                               Ce0[5],0,0,0,
                                                               Ce0[6],0,0,
                                                               Ce0[7],0,
                                                               Ce0[8]])
pr.matlID = end_prop
pr.Put(end_prop)


elem=app.feElem
nel=len(elements)//20
print(nel,eln)
print(elem.PutAllArray(nel,np.arange(1,nel+1),prop_id,[25]*nel,[8]*nel,[1]*nel,[124]*nel,[0]*2*nel,[0]*3*nel,[0]*nel*6,[0]*nel*12,[0]*nel,[0]*nel,elements,[0]*2*nel,[0]*2*nel))







