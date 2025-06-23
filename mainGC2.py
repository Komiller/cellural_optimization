import time
from collections import deque

import numpy as np

from model import model
import matplotlib.pyplot as plt
from visual_utils import dump_new,write_data_on_hexmesh,dump_new2
import os
from datetime import datetime
from MMA import mmasub, gcmmasub, asymp
from scipy.sparse.linalg import spsolve
from icecream import ic
from problem import Problem
from modelling_utils import *

alpha=1.004
beta=0.05
def update_constr(u,U,**kwarg):
    if kwarg.get('flag'):
        Ui = (u + beta*u)*1.2
    elif u<=alpha*U:
        Ui= u + min(beta*u,(alpha*U-u))
    else:
        Ui= u - min(beta*u,abs(alpha*U-u))
    return Ui

folder='optimization_results/opt_res_2025-06-09_21-07-52'
def main(nelx,nely,nelz,volfrac,rmin,rstart,maxloop,commentary,mmmove,target_volf,is_gc):

    ndof = 3 * (nelx + 1) * (nely + 1) * (nelz + 1)
    """setting initial design and optimization domain"""
    specific_domain=False
    #x=np.random.beta(8,2,size=3*nely * nelx * nelz)*0.49
    x = np.ones(3*nely * nelx * nelz, dtype=float)*rstart
    #x = np.load('optimization_results/opt_res_2025-06-07_14-29-26/radii_100.npy').flatten()


    if specific_domain:
        xmask = np.zeros(nelx*nely*nelz,dtype=bool)
        for xx in range(nelx):
            for yy in range(nely):
                for zz in range(nelz):
                    center = (xx + 0.5, yy + 0.5,zz+0.5)
                    idx = xx + nelx * yy + nelx * nely * zz
                    xmask[idx] = profile_cyllinder(center,nelx,nely)
    else: xmask = None
    # x[np.kron(xmask,np.ones(3)).flatten().astype(bool)]=x1




    # BC's and support
    [jf, kf] = np.meshgrid(np.arange(nelx + 1), np.arange(nely + 1))  # Coordinates
    fixednid1 = (kf) * (nelx + 1) + jf
    fixed1 = 3*fixednid1.flatten()+2

    fixednid2 = (kf) * (nelx + 1) + jf
    fixed2 = np.kron(3 * fixednid2.flatten(), np.ones(3)).astype(np.int32)
    fixed2[1::3] += 1
    fixed2[2::3] += 2

    # fixed = np.concatenate([fixed1,fixed2])
    fixed = fixed2

    dofs = np.arange(3 * (nelx + 1) * (nely + 1) * (nelz + 1))
    free = np.setdiff1d(dofs, fixed)

    # Solution and RHS vectors
    f = np.zeros((ndof, 1))
    u = np.zeros((ndof, 1))

    # Load

    loadnid2 = (kf) * (nelx + 1) + jf +(nelx+1)*(nely+1)*nelz  # Node IDs
    loaddof21 = 3 * loadnid2.flatten().astype(np.int32) + 1  # DOFs
    loaddof22 = 3 * loadnid2.flatten().astype(np.int32)[[0,-1]] + 0  # DOFs
    f[loaddof21, 0] = 1
    # f[loaddof22, 0] = 0.5
    # fc=f_c(nelx,nely).flatten()
    # f[loaddof21, 0] = fc[::3]
    # f[loaddof22, 0] = fc[1::3]

    # outer_nodes_arr = outer_nodes(nelx, nely, nelz, xmask)
    # f = apply_presure_load_on_cyllinder(outer_nodes_arr, nelx, nely, nelz).reshape(len(f),1)

    plot_disp(f, fixed, nelx, nely, nelz)

    f = np.load(f'{folder}/force.npy')
    free = np.load(f'{folder}/bc_free.npy')
    """setting mask for loaded units"""
    fmask=np.zeros(3*nelx*nely*nelz,dtype=bool)
    for xx in range(nelx):
        for yy in range(nely):
            for zz in range(nelz):
                idx = xx + nelx * yy + nelx * nely * zz
                nodes_arr=np.array([xx + (nelx+1) * yy + (nelx+1) * (nely+1) * zz,xx+1 + (nelx+1) * yy + (nelx+1) * (nely+1) * zz,
                           xx + (nelx+1) * (yy+1) + (nelx+1) * (nely+1) * zz,xx + (nelx+1) * yy + (nelx+1) * (nely+1) * (zz+1),
                           xx + 1 + (nelx+1) * (yy+1) + (nelx+1) * (nely+1) * zz,xx + (nelx+1) * (yy+1) + (nelx+1) * (nely+1) * (zz+1),
                           xx + 1 + (nelx+1) * yy + (nelx+1) * (nely+1) * (zz+1),xx +1 + (nelx+1) * (yy+1) + (nelx+1) * (nely+1) * (zz+1)],dtype=np.int32)
                if np.abs(f[3*nodes_arr].sum())+np.abs(f[3*nodes_arr+1].sum())+np.abs(f[3*nodes_arr+2].sum()) != 0 : fmask[3*idx:3*idx+3]=True

    if specific_domain:
        fmask = fmask[np.kron(xmask, np.ones(3)).flatten().astype(bool)]

    """displacement constraints"""
    disp_constr = (kf[nely // 2, :]) * (nelx + 1) + jf[nely // 2, :] + (nelx + 1) * (nely + 1) * nelz
    # dispconstrdofs = 3 * np.concatenate([disp_constr[:1], disp_constr[-1:]]) + 1
    # dispconstrdofs = 3*disp_constr[nelx//2-1:nelx//2+2]+1
    # dispconstrdofs =  np.array([3*((kf[nely//2, 0]) * (nelx + 1) + jf[nely//2,0])])
    dispconstrdofs = np.array([],dtype=np.int32)
    dispconstrval = 0.4 * np.ones(len(dispconstrdofs))
    




   
    


    
    """setting up problem"""
    my_problem = Problem(nelx, nely, nelz, rmin, target_volf, x, f, u, free, weight_path='weights', scaler_path='scalers' \

                         , comment=commentary, constrainedDofs=dispconstrdofs, disp_constr_value=dispconstrval,
                         xmask=xmask, fixedr=fmask)

    #regularization coefficient for disp constr sensitivity
    my_problem.regc=0.9
    #penalization coefficient
    my_problem.pp=1
    my_problem.volfrac=target_volf

    rod_mesh_path=os.path.join(my_problem.log_dir,'rod_mesh.vtu')
    box_mesh_path=os.path.join(my_problem.log_dir,'box_mesh.vtu')
    
    del x,f,u,free

    loop = 0
    change = 1
    t1 = 0
    # Initialize MMA
    if specific_domain:n = 3 * xmask.sum()
    else:n = 3*nelx*nely*nelz
    m = 1 + len(my_problem.constrainedDofs)
    xmin = np.zeros((n, 1))
    xmin[fmask] = 0.1
    xmax = np.ones((n, 1)) * 0.5
    xval = my_problem.x[np.newaxis].T
    xold1 = xval.copy()
    xold2 = xval.copy()
    low = np.ones((n, 1))
    upp = np.ones((n, 1))
    a0 = 1.0
    a = np.zeros((m, 1))
    c = 10000 * np.ones((m, 1))
    d = np.zeros((m, 1))
    move = mmmove
    albefa = 0.1

    epsimin = 1e-7
    raa0 = 0.01
    raa = 0.01 * np.zeros((m,1))
    raa0eps = 1e-6
    raaeps=1e-6 * np.zeros((m,1))

    constr = 1
    obj = 1

    constr_queue = deque(maxlen=5)
    constr_queue.append(100)
    disp_constr_queue = deque(maxlen=5)
    disp_constr_queue.append(0)
    my_problem.volfrac=0
    print(dispconstrval)
    my_problem.disp_constr_val[:]=0
    
    while (change > 0.0001 or loop<10) and loop < maxloop:

        if constr < 0.01 and (my_problem.volfrac - 0.009) > target_volf: my_problem.volfrac -= 0.01
        
        if loop == 60: my_problem.pp = 1
        print('vv', my_problem.volfrac)

        if loop % 5 == 0:
            if specific_domain:
                x_dump = np.zeros(3 * nelx * nely * nelz)
                x_dump[np.kron(xmask, np.ones(3)).flatten().astype(bool)] = my_problem.xPhys.flatten()
                dump_new(x_dump.flatten(), rod_mesh_path, nelx, nely, nelz)
            else: dump_new(my_problem.xPhys.flatten(), rod_mesh_path, nelx, nely, nelz)

        



        t = time.perf_counter()
        t1old = t1
        t1 = time.time()


        obj=my_problem.obj_func()
        constr_disp=my_problem.constraintDisp()
        constr=my_problem.constraintvoluem()
        cdr= 0 #((constr_disp-dispconstrval)/dispconstrval).max()

        flag=constr-target_volf>=0.005
        constr=constr-update_constr(constr,target_volf)
        
        for i in range(len(constr_disp)):
            constr_disp[i]=constr_disp[i]-update_constr(constr_disp[i],dispconstrval[i])

        print('disp_constr',constr_disp)
        if obj is np.nan:
            if specific_domain:
                rho = np.zeros(nelx * nely * nelz)
                rho[xmask] = my_problem.rho
                write_data_on_hexmesh(box_mesh_path, nelx + 1, nely + 1, nelz + 1, rho=rho)
            else: write_data_on_hexmesh(box_mesh_path, nelx + 1, nely + 1, nelz + 1, rho=my_problem.rho)
        elif obj<0 or obj>100000 or loop==1:
            if specific_domain:
                rho = np.zeros(nelx * nely * nelz)
                rho[xmask] = my_problem.rho
                cell_energy1 = (1 / 2 * my_problem.sK.flatten() * my_problem.u[my_problem.jK].flatten() * my_problem.u[
                    my_problem.iK].flatten()).reshape(len(my_problem.x) // 3, 576).sum(1)
                cell_energy = np.zeros(nelx*nely*nelz)
                cell_energy[xmask] = cell_energy1
                write_data_on_hexmesh(box_mesh_path, nelx + 1, nely + 1, nelz + 1, rho=rho,energy=cell_energy)
            else: 
                cell_energy = (1 / 2 * my_problem.sK.flatten() * my_problem.u[my_problem.jK].flatten() * my_problem.u[
                    my_problem.iK].flatten()).reshape(len(my_problem.x) // 3, 576).sum(1)
                write_data_on_hexmesh(box_mesh_path, nelx + 1, nely + 1, nelz + 1, rho=my_problem.rho,energy=cell_energy)


        constr_queue.append(max(constr, 0))
        if len(dispconstrdofs)>0: disp_constr_queue.append(np.max(np.clip(constr_disp, a_min=0, a_max=None)))
        
        print('tt ',time.time()-t1)

        print('constr',constr)
        # MMA step
        if sum(disp_constr_queue) > 0 and sum(constr_queue) > 0:
            mu0 = 5
            mu1 = 2
            mu2 = 0.1
        elif sum(disp_constr_queue) > 0 and sum(constr_queue) <= 0:
            mu0 = 5
            mu1 = 1
            if cdr > 2:
                mu2=2
            else: mu2 = 5
            
        elif sum(disp_constr_queue) <= 0 and sum(constr_queue) > 0:
            mu0 = 5
            mu1 = 2
            mu2 = 0.05
        elif sum(disp_constr_queue) <= 0 and sum(constr_queue) <= 0:
            mu0 = 10
            mu1 = 1
            mu2 = 1

        mu0 *= 0.5
        mu1 *= 0.5
        mu2 *= 2

        f0val = mu0 * obj
        df0dx = mu0 * my_problem.grad_obj[np.newaxis].T / np.abs(my_problem.grad_obj).max()

        clip_val = 1 #np.abs(my_problem.grad_constr_disp).max() #np.percentile(np.abs(my_problem.grad_constr_disp), 97)
        if len(dispconstrdofs)>0:my_problem.grad_constr_disp = np.clip(my_problem.grad_constr_disp, a_min=-clip_val, a_max=clip_val)
        fval = np.hstack((mu1 * constr, mu2 * constr_disp/clip_val)).reshape((m, 1))
        dfdx = np.vstack((mu1 * my_problem.grad_constr_vol[np.newaxis], mu2 * my_problem.grad_constr_disp/clip_val))

        xval = my_problem.x.copy()[np.newaxis].T
        k = 399
        
        
        if is_gc:
            low,upp,raa0,raa = asymp(k,n,xval,xold1,xold2,xmin,xmax,low,upp,raa0,raa,raa0eps,raaeps,df0dx,dfdx)
            xmma,ymma,zmma,lam,xsi,eta,mu,zet,s,f0app,fapp = \
            gcmmasub(m,n,k,epsimin,xval,xmin,xmax,low,upp,raa0,raa,f0val,df0dx,fval,dfdx,a0,a,c,d,albefa)
        else:
            xmma, ymma, zmma, lam, xsi, eta, mu, zet, s, low, upp = \
                mmasub(m, n, k, xval, xmin, xmax, xold1, xold2, f0val, df0dx, fval, dfdx, low, upp, a0, a, c, d, move)

        


        xold2 = xold1.copy()
        xold1 = xval.copy()
        my_problem.x = xmma.copy().flatten()

        # Compute the change by the inf. norm
        change = 1 # np.linalg.norm(my_problem.x.reshape(3*nelx * nely * nelz, 1) - xold1.reshape(3*nelx * nely * nelz, 1), np.inf)
        loop = loop + 1
        t3 = time.time()

        t4 = time.time()

    print('finwolf', my_problem.volfrac)

    if specific_domain:
        rho = np.zeros(nelx * nely * nelz)
        rho[xmask] = my_problem.rho
        df = np.zeros(nelx * nely * nelz)
        df[xmask] = fmask[::3]
        Th = np.zeros(nelx * nely * nelz)
        Th[xmask] = my_problem.Th.flatten()
        rho = np.zeros(nelx * nely * nelz)
        rho[xmask] = my_problem.rho
        write_data_on_hexmesh(box_mesh_path, nelx + 1, nely + 1, nelz + 1, Th=Th, rho=rho, df=df,
                          xmask=xmask.astype(np.int32))
        x_dump = np.zeros(3 * nelx * nely * nelz)
        x_dump[np.kron(xmask, np.ones(3)).flatten().astype(bool)] = my_problem.xPhys.flatten()
        dump_new(x_dump.flatten(), rod_mesh_path, nelx, nely, nelz)
    else:
        cell_energy = (1 / 2 * my_problem.sK.flatten() * my_problem.u[my_problem.jK].flatten() * my_problem.u[
                    my_problem.iK].flatten()).reshape(len(my_problem.x) // 3, 576).sum(1)
        write_data_on_hexmesh(box_mesh_path, nelx + 1, nely + 1, nelz + 1, Th=my_problem.Th, rho=my_problem.rho,energy = cell_energy)

    






if __name__ == '__main__':
    nelx=10
    nely=10
    nelz=20
    commentary='box, irregularity tests'
    for target_volf in [0.2]:
        main(nelx,nely,nelz,0.1,1.4,0.2,100,commentary,0.2,target_volf,True)
