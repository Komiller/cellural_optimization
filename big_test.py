import time
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import cg
import pyamg
import numpy as np
from model import model
from main import plot_disp
from visual_utils import write_data_on_hexmesh
from visual_utils import dump, dump_new2,dump_new
from scipy.linalg import solve
from problem import Problem
from modelling_utils import *

nelx=10
nely=10
nelz=50
volfrac=0.14
rstart=0.2
rmin=1.5
target_volf=0.1
commentary='1'
folder='optimization_results/another_cell/bending'
ndof = 3 * (nelx + 1) * (nely + 1) * (nelz + 1)
"""setting initial design and optimization domain"""
specific_domain = True
# x=np.random.beta(8,2,size=3*nely * nelx * nelz)*0.49
x = np.ones(3 * nely * nelx * nelz, dtype=float) * rstart
x1 = np.load(f'{folder}/radii_100.npy').flatten()
xmask=np.load(f'{folder}/domain.npy').flatten()
write_data_on_hexmesh('wing_meshbt.vtu',nelx+1,nely+1,nelz+1,Th=xmask.astype(float))
print(1)
# if specific_domain:
#     xmask = np.zeros(nelx * nely * nelz, dtype=bool)
#     for xx in range(nelx):
#         for yy in range(nely):
#             for zz in range(nelz):
#                 center = (xx + 0.5, yy + 0.5, zz + 0.5)
#                 idx = xx + nelx * yy + nelx * nely * zz
#                 xmask[idx] = profile_cyllinder(center, nelx, nely)
# else:
#     xmask = None
x[np.kron(xmask,np.ones(3)).flatten().astype(bool)]=x1


# BC's and support
[jf, kf] = np.meshgrid(np.arange(nelx + 1), np.arange(nely + 1))  # Coordinates
# fixednid1 = (kf) * (nelx + 1) + jf
# fixed1 = 3*fixednid1.flatten()+2
#
# fixednid2 = (kf) * (nelx + 1) + jf
# fixed2 = np.kron(3 * fixednid2.flatten(), np.ones(3)).astype(np.int32)
# fixed2[1::3] += 1
# fixed2[2::3] += 2
#
# # fixed = np.concatenate([fixed1,fixed2])
# fixed = fixed2
#
# dofs = np.arange(3 * (nelx + 1) * (nely + 1) * (nelz + 1))
# free = np.setdiff1d(dofs, fixed)

# Solution and RHS vectors

u = np.zeros((ndof, 1))

# Load





# fc=f_c(nelx,nely).flatten()
# f[loaddof21, 0] = fc[::3]
# f[loaddof22, 0] = fc[1::3]

# outer_nodes_arr = outer_nodes(nelx, nely, nelz, xmask)
# f = apply_presure_load_on_prof0012(outer_nodes_arr, nelx, nely, nelz,60,35).reshape(len(f),1)

# plot_disp(f, fixed, nelx, nely, nelz)


f=np.load(f'{folder}/force.npy')
free=np.load(f'{folder}/bc_free.npy')
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




# plot_disp(f,free,nelx,nely,nelz)

if specific_domain:
    fmask = fmask[np.kron(xmask, np.ones(3)).flatten().astype(bool)]

"""displacement constraints"""
disp_constr = (kf[nely // 2, :]) * (nelx + 1) + jf[nely // 2, :] + (nelx + 1) * (nely + 1) * nelz
# dispconstrdofs = 3 * np.concatenate([disp_constr[:1], disp_constr[-1:]]) + 1
# dispconstrdofs = 3*disp_constr[nelx//2-1:nelx//2+2]+1
dispconstrdofs = np.array([3 * ((kf[nely // 2, 0]) * (nelx + 1) + jf[nely // 2, 0])])
# dispconstrdofs = np.array([],dtype=np.int32)
dispconstrval = 0.4 * np.ones(len(dispconstrdofs))

"""setting up problem"""
my_problem = Problem(nelx, nely, nelz, rmin, target_volf, x, f, u, free, weight_path='weights', scaler_path='scalers' \
 \
                     , comment=commentary, constrainedDofs=dispconstrdofs, disp_constr_value=dispconstrval,
                     xmask=xmask, fixedr=fmask,log_status=False)

obj = my_problem.obj_func()
print(obj)

# print(my_problem.rho.sum()/(nelx*nely*nelz))
# dump(my_problem.xPhys.flatten()*np.kron(my_problem.Th,np.ones(3)).flatten(),'model1',nelx,nely,nelz)
# np.save('u_test_bending',my_problem.u)
# print(obj)
# cell_energy1 = (1 / 2 * my_problem.sK.flatten() * my_problem.u[my_problem.jK].flatten() * my_problem.u[my_problem.iK].flatten()).reshape(len(my_problem.x) // 3, 576).sum(1)
# cell_energy=np.zeros(len(x)//3)
# cell_energy[xmask]=cell_energy1
#
# u=np.zeros(3*(nelx+1)*(nely+1)*(nelz+1))
# u[my_problem.dof_mask]=my_problem.u.flatten()
# np.set_printoptions(linewidth=np.inf,precision=3,suppress=True,threshold=np.inf)
# print(u[3*((nelx+1)*(nely+1)*nelz)+1::3].reshape((nelx+1,nely+1)))
# print(my_problem.rho.sum()/(my_problem.n/3))
#
# maxlx=nelz
# x_dump = np.zeros(3 * nelx * nely * nelz)
# x_dump[np.kron(xmask,np.ones(3)).flatten().astype(bool)]=my_problem.xPhys.flatten()
# t1=time.time()
# dump(x_dump, 'testbt1',nelx,nely,nelz)
# print('dump_time',time.time()-t1)
# print('start')
# Th=np.zeros(nelx * nely * nelz)
# Th[xmask]=my_problem.Th.flatten()
# rho=np.zeros(nelx * nely * nelz)
# rho[xmask]=my_problem.rho
# print('cc',cell_energy1.sum())
# print('cc',cell_energy1[cell_energy1>0].sum())
# write_data_on_hexmesh('wing_meshbt.vtu',nelx+1,nely+1,nelz+1,Th=Th,r1=x_dump[::3],r2=x_dump[1::3],r3=x_dump[2::3],energy=cell_energy,
#                       rho=rho) #sensr1=my_problem.grad_obj[0::3],sensr2=my_problem.grad_obj[1::3],sensr3=my_problem.grad_obj[2::3]
# print('Th.sum() ',Th.sum())
# x_dump = np.zeros(3 * nelx * nely * nelz)
# x_dump[np.kron(xmask,np.ones(3)).flatten().astype(bool)] = my_problem.xPhys.flatten()*np.kron(my_problem.Th,np.ones(3)).flatten()
# x_dump=x_dump
# t1=time.time()
# dump(x_dump.flatten(), 'test1',nelx,nely,nelz)
# print('dump_time',time.time()-t1)
# print('start')
# Th=np.zeros(nelx * nely * nelz)
# Th[xmask]=my_problem.Th.flatten()
# write_data_on_hexmesh('wing_mesh.vtu',nelx+1,nely+1,nelz+1,Th=Th,r1=x_dump[::3],r2=x_dump[1::3],r3=x_dump[2::3])


#
# dump(my_problem.xPhys*np.kron(my_problem.Th,np.ones(3)).flatten(),'test1',nelx,nely, nelz)
# u=np.zeros(3*(nelx+1)*(nely+1)*(nelz+1))
# u=my_problem.u.flatten()
# np.set_printoptions(linewidth=np.inf,precision=3,suppress=True,threshold=np.inf)
# print(u[3*((nelx+1)*(nely+1)*nelz)+1::3].reshape((nelx+1,nely+1)))
# write_data_on_hexmesh('wing_mesh.vtu',nelx+1,nely+1,nelz+1,Th=my_problem.Th,r1=my_problem.xPhys[::3],r2=my_problem.xPhys[1::3],r3=my_problem.xPhys[2::3],
#                       rho=my_problem.rho)


