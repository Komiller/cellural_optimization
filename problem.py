import time

import numpy as np
from numpy.typing import NDArray
from model import model,isotropic
from scipy.sparse import coo_matrix
from scipy.linalg import solve
from scipy.sparse.linalg import cg
import pyamg
from utilities import Logger

class Problem(Logger):
    def __init__(self,nelx: int,nely: int,nelz: int,rmin: float,volfrac: float,x:  NDArray[np.float64],
                 f:  NDArray[np.float64],u:  NDArray[np.float64],free:  NDArray[np.int32], constrainedDofs = np.zeros(0), disp_constr_value = np.zeros(0),
                 comment=None,rstart=None,weight_path='weights', scaler_path='scalers',hx=1,hy=1,hz=1,xmask=None,log_status=True,fixedr=None):
        """
        :param nelx: elements on x
        :param nely: elements on y
        :param nelz: elements on z
        :param rmin: filter radius
        :param volfrac: volume constraint
        :param x: project point
        :param f: forces
        :param u: displacements
        :param free: free dof's
        """
        if rstart is None: rstart=x[0]
        super().__init__(log_status,'optimization_results',nelx=nelx,nely=nely,nelz=nelz,rmin=rmin,volfrac=volfrac,rstart=rstart,commentary=comment,
                         bc_free=free,force=f,domain=xmask)
        if log_status: self.set_log_objects()

        self.nelx = nelx
        self.nely = nely
        self.nelz = nelz
        self.hx = hx
        self.hy = hy
        self.hz = hz
        self.rmin = rmin
        self.volfrac = volfrac

        self.n = len(x)
        self.ndof = 3 * (nelx + 1) * (nely + 1) * (nelz + 1)
        self.x=x
        self.xPhys=x.copy()

        self.grad_obj=np.zeros(self.n)
        self.grad_constr_vol = np.zeros(self.n)

        self.f=f
        self.u=u
        self.free=free

        self._build_filter_matrix_()
        self._build_index_vectors_()
        self._init_shape_function_()
        self.elasticity_model = model(weight_path,scaler_path)

        self.constrainedDofs = constrainedDofs.copy()
        self.disp_constr_val = disp_constr_value.copy()
        self.grad_constr_disp=np.zeros((len(self.constrainedDofs),self.n))

        self.K=0
        self.K_diffs=0
        self.regc=0.9


        self.penal=3
        if xmask is not None:
            self.xmask=xmask
            self._update_problem_on_mask()

        self.fixedr=fixedr
        self.pp=2



    def _update_problem_on_mask(self):
        self.dof_mask=np.zeros(self.ndof,dtype=bool)
        """degrees of freedom mask intialisation"""
        for xx in range(self.nelx + 1):
            for yy in range(self.nely + 1):
                for zz in range(self.nelz + 1):
                    idx = xx + (self.nelx + 1) * yy + (self.nelx + 1) * (self.nely + 1) * zz
                    adj_elem = []
                    if xx < self.nelx and yy < self.nely and zz < self.nelz:
                        adj_elem.append(xx + (self.nelx) * yy + (self.nelx) * (self.nely) * zz)

                    if xx < self.nelx and yy < self.nely and zz > 0:
                        adj_elem.append(xx + (self.nelx) * yy + (self.nelx) * (self.nely) * (zz - 1))
                    if xx < self.nelx and yy > 0 and zz < self.nelz:
                        adj_elem.append(xx + (self.nelx) * (yy - 1) + (self.nelx) * (self.nely) * zz)
                    if xx > 0 and yy < self.nely and zz < self.nelz:
                        adj_elem.append((xx - 1) + (self.nelx) * (yy) + (self.nelx) * (self.nely) * zz)

                    if xx < self.nelx and yy > 0 and zz > 0:
                        adj_elem.append(xx + (self.nelx) * (yy - 1) + (self.nelx) * (self.nely) * (zz - 1))
                    if xx > 0 and yy < self.nely and zz > 0:
                        adj_elem.append((xx - 1) + (self.nelx) * yy + (self.nelx) * (self.nely) * (zz - 1))
                    if xx > 0 and yy > 0 and zz < self.nelz:
                        adj_elem.append((xx - 1) + (self.nelx) * (yy - 1) + (self.nelx) * (self.nely) * zz)

                    if xx > 0 and yy > 0 and zz > 0:
                        adj_elem.append((xx - 1) + (self.nelx) * (yy - 1) + (self.nelx) * (self.nely) * (zz - 1))
                    number = bool(self.xmask[adj_elem].sum())
                    self.dof_mask[3 * idx] = number
                    self.dof_mask[3 * idx + 1] = number
                    self.dof_mask[3 * idx + 2] = number

        renumber_array=np.zeros(self.ndof).astype(np.int32)
        renumber_array[self.dof_mask]=np.arange(self.dof_mask.sum())
        tmp=np.kron(self.xmask,np.ones(3)).flatten().astype(bool)
        """udating problem"""
        self.n = 3*self.xmask.sum()
        self.ndof = self.dof_mask.sum()
        self.x = self.x[tmp]
        self.xPhys = self.x.copy()

        self.grad_obj = np.zeros(self.n)
        self.grad_constr_vol = np.zeros(self.n)

        self.f = self.f[self.dof_mask,:]
        self.u = self.u[self.dof_mask,:]
        self.free = renumber_array[self.free[self.dof_mask[self.free]]]

        self.H = self.H[self.xmask, :][:, self.xmask]
        self.Hs = self.H.sum(1)


        tmp = np.kron(self.xmask, np.ones(24*24,np.int8)).flatten().astype(bool)
        self.iK = renumber_array[self.iK[tmp]]
        self.jK = renumber_array[self.jK[tmp]]

        self.constrainedDofs = renumber_array[self.constrainedDofs]
        self.grad_constr_disp = np.zeros((len(self.constrainedDofs), self.n))

    def set_log_objects(self):
        self._set_log_object('x',False,10,'radii')

    def _build_filter_matrix_(self):
        nfilter = int(self.nelx * self.nely * self.nelz * ((2 * (np.ceil(self.rmin) - 1) + 1) ** 3))
        iH = np.zeros(nfilter)
        jH = np.zeros(nfilter)
        sH = np.zeros(nfilter)
        cc = 0
        for z in range(self.nelz):
            for i in range(self.nelx):
                for j in range(self.nely):
                    row = i * self.nely + j + z * (self.nelx * self.nely)
                    kk1 = int(np.maximum(i - (np.ceil(self.rmin) - 1), 0))
                    kk2 = int(np.minimum(i + np.ceil(self.rmin), self.nelx))
                    ll1 = int(np.maximum(j - (np.ceil(self.rmin) - 1), 0))
                    ll2 = int(np.minimum(j + np.ceil(self.rmin), self.nely))
                    mm1 = int(np.maximum(z - (np.ceil(self.rmin) - 1), 0))
                    mm2 = int(np.minimum(z + np.ceil(self.rmin), self.nelz))
                    for mm in range(mm1, mm2):
                        for kk in range(kk1, kk2):
                            for ll in range(ll1, ll2):
                                col = kk * self.nely + ll + mm * (self.nelx * self.nely)
                                fac = self.rmin - np.sqrt((i - kk) * (i - kk) + (j - ll) * (j - ll) + (z - mm) * (z - mm))
                                iH[cc] = row
                                jH[cc] = col
                                sH[cc] = np.maximum(0.0, fac)
                                cc = cc + 1
        # Finalize assembly and convert to csc format
        self.H = coo_matrix((sH, (iH, jH)), shape=(self.nelx * self.nely * self.nelz, self.nelx * self.nely * self.nelz)).tocsc()
        self.Hs = self.H.sum(1)

    def _build_index_vectors_(self):
        edofMat = np.zeros((self.nelx * self.nely * self.nelz, 24), dtype=np.int32)
        for elz in range(self.nelz):
            for ely in range(self.nely):
                for elx in range(self.nelx):
                    el = elx + (ely * self.nelx) + elz * (self.nelx * self.nely)
                    n1 = elx + ely * (self.nelx + 1) + elz * (self.nelx + 1) * (self.nely + 1)
                    n2 = elx + (ely + 1) * (self.nelx + 1) + elz * (self.nelx + 1) * (self.nely + 1)
                    n3 = elx + ely * (self.nelx + 1) + (elz + 1) * (self.nelx + 1) * (self.nely + 1)
                    n4 = elx + (ely + 1) * (self.nelx + 1) + (elz + 1) * (self.nelx + 1) * (self.nely + 1)
                    edofMat[el, :] = np.array(
                        [3 * n1, 3 * n1 + 1, 3 * n1 + 2, 3 * n1 + 3, 3 * n1 + 4, 3 * n1 + 5,
                         3 * n2 + 3, 3 * n2 + 4, 3 * n2 + 5, 3 * n2, 3 * n2 + 1, 3 * n2 + 2,
                         3 * n3, 3 * n3 + 1, 3 * n3 + 2, 3 * n3 + 3, 3 * n3 + 4, 3 * n3 + 5,
                         3 * n4 + 3, 3 * n4 + 4, 3 * n4 + 5, 3 * n4, 3 * n4 + 1, 3 * n4 + 2])

        # Construct the index pointers for the coo format
        self.iK = np.kron(edofMat, np.ones((24, 1),dtype=np.int32)).flatten().astype(np.int32)
        self.jK = np.kron(edofMat, np.ones((1, 24),dtype=np.int32)).flatten().astype(np.int32)

    def _init_shape_function_(self):
        """Three Gauss points in both directions"""
        gaussian_points = [-np.sqrt(3 / 5), 0, np.sqrt(3 / 5)]
        weights = [5 / 9, 8 / 9, 5 / 9]
        Bs = []
        BTs = []
        weight_arr = []
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    """integration point"""
                    x = gaussian_points[i]
                    y = gaussian_points[j]
                    z = gaussian_points[k]

                    """stress strain displacement matrix"""
                    qx = [-((y - 1) * (z - 1)) / 8, ((y - 1) * (z - 1)) / 8, -((y + 1) * (z - 1)) / 8,
                          ((y + 1) * (z - 1)) / 8, ((y - 1) * (z + 1)) / 8, -((y - 1) * (z + 1)) / 8,
                          ((y + 1) * (z + 1)) / 8, -((y + 1) * (z + 1)) / 8]

                    qy = [-((x - 1) * (z - 1)) / 8, ((x + 1) * (z - 1)) / 8, -((x + 1) * (z - 1)) / 8,
                          ((x - 1) * (z - 1)) / 8, ((x - 1) * (z + 1)) / 8, -((x + 1) * (z + 1)) / 8,
                          ((x + 1) * (z + 1)) / 8, -((x - 1) * (z + 1)) / 8]

                    qz = [-((x - 1) * (y - 1)) / 8, ((x + 1) * (y - 1)) / 8, -((x + 1) * (y + 1)) / 8,
                          ((x - 1) * (y + 1)) / 8, ((x - 1) * (y - 1)) / 8, -((x + 1) * (y - 1)) / 8,
                          ((x + 1) * (y + 1)) / 8, -((x - 1) * (y + 1)) / 8]

                    """Jacobian"""
                    J1 = np.matrix([qx, qy, qz])
                    arr1 = np.array(
                        [-0.5 * self.hx, 0.5 * self.hx, 0.5 * self.hx, -0.5 * self.hx, -0.5 * self.hx, 0.5 * self.hx, 0.5 * self.hx, -0.5 * self.hx])
                    arr2 = np.array(
                        [-0.5 * self.hy, -0.5 * self.hy, 0.5 * self.hy, 0.5 * self.hy, -0.5 * self.hy, -0.5 * self.hy, 0.5 * self.hy, 0.5 * self.hy])
                    arr3 = np.array(
                        [-0.5 * self.hz, -0.5 * self.hz, -0.5 * self.hz, -0.5 * self.hz, 0.5 * self.hz, 0.5 * self.hz, 0.5 * self.hz, 0.5 * self.hz])
                    J2 = np.matrix([arr1, arr2, arr3]).T
                    J = J1 * J2
                    qxyz = solve(J, J1)

                    "qxyz fix loop"
                    for a in range(0, 8, 2):
                        temporary_variable = qxyz[1][a]
                        qxyz[1][a] = qxyz[1][a + 1]
                        qxyz[1][a + 1] = temporary_variable

                        temporary_variable = qxyz[2][a]
                        qxyz[2][a] = qxyz[2][a + 1]
                        qxyz[2][a + 1] = temporary_variable

                    B_e = np.zeros((6, 3, 8))

                    for i_B in range(0, 8):
                        B_e[:, :, i_B] = np.matrix([[qxyz[0, i_B], 0, 0], [0, qxyz[1, i_B], 0], [0, 0, qxyz[2, i_B]],
                                                    [qxyz[1, i_B], qxyz[0, i_B], 0], [0, qxyz[2, i_B], qxyz[1, i_B]],
                                                    [qxyz[2, i_B], 0, qxyz[0, i_B]]])

                    B = np.concatenate(
                        (B_e[:, :, 0], B_e[:, :, 1], B_e[:, :, 2], B_e[:, :, 3], B_e[:, :, 4], B_e[:, :, 5],
                         B_e[:, :, 6],
                         B_e[:, :, 7]), axis=1)

                    """Weight factor at this point"""
                    weight = np.linalg.det(J) * weights[i] * weights[j] * weights[k]
                    Bs.append(B)
                    BTs.append(B.T)
                    weight_arr.append(weight)

        self.Bs=Bs
        self.BTs=BTs
        self.weights=weight_arr

    def filter_x(self,x=None, grad=None):
        self.xPhys = self.x.copy()
        """applying filter"""
        for i in range(3):
            self.xPhys[i::3] = np.asarray(self.H * self.x[i::3][np.newaxis].T / self.Hs)[:, 0]

        self.rho = self.elasticity_model.predict_value_density(self.xPhys[0::3], self.xPhys[1::3],
                                                               self.xPhys[2::3]).flatten()
        a = 1e4
        b = 0.1
        Th = np.array(list(map(lambda x: 1 / 2 * (1 + np.tanh(a * (x - b))), self.rho))).flatten()
        Th[self.fixedr[0::3]] = 1

        minimalr = 0.1
        self.xPhys[np.logical_and(np.kron(Th, np.ones(3)).flatten() > 0.1, self.xPhys < minimalr)] = minimalr
        if self.fixedr is not None:
            self.xPhys[self.fixedr] = np.clip(self.xPhys[self.fixedr], a_min=0.1, a_max=None)
    @Logger.log_sub('Obj')
    def obj_func(self,x=None, grad=None):
        '''
        function for calculation of object function and sensitives
        :param x:
        :param grad:
        :return:
        '''
        if x is not None: self.x=x





        self.xPhys = self.x.copy()
        """applying filter"""
        for i in range(3):
            self.xPhys[i::3] = np.asarray(self.H * self.x[i::3][np.newaxis].T / self.Hs)[:, 0]

        self.rho = self.elasticity_model.predict_value_density(self.xPhys[0::3], self.xPhys[1::3], self.xPhys[2::3]).flatten()
        a = 1e4
        b = 0.1
        Th = np.array(list(map(lambda x: 1 / 2 * (1 + np.tanh(a * (x - b))), self.rho))).flatten()
        Th[self.fixedr[0::3]]=1

        minimalr=0.1
        self.xPhys[np.logical_and(np.kron(Th, np.ones(3)).flatten() > 0.1, self.xPhys < minimalr)] = minimalr
        if self.fixedr is not None:
            self.xPhys[self.fixedr]=np.clip(self.xPhys[self.fixedr],a_min=0.1,a_max=None)


        self.rho = self.elasticity_model.predict_value_density(self.xPhys[0::3], self.xPhys[1::3],
                                                               self.xPhys[2::3]).flatten()



        self.Th=Th
        self.rho = self.rho * Th
        # xPhys = xPhys * np.kron(Th, np.ones(3)).flatten()

        """build and assamble global stiffnes matrix and its sensetivites"""

        t1 = time.time()
        h=0.01
        C = self.elasticity_model.predict_value_CE(self.xPhys[0::3], self.xPhys[1::3], self.xPhys[2::3])
        # C1 = self.elasticity_model.predict_value_CE(self.xPhys[0::3]+h, self.xPhys[1::3], self.xPhys[2::3])
        # C2 = self.elasticity_model.predict_value_CE(self.xPhys[0::3], self.xPhys[1::3]+h, self.xPhys[2::3])
        # C3 = self.elasticity_model.predict_value_CE(self.xPhys[0::3], self.xPhys[1::3], self.xPhys[2::3]+h)
        C_diffs = self.elasticity_model.predict_sensitivies_CE(self.xPhys[0::3], self.xPhys[1::3], self.xPhys[2::3])* np.kron(np.kron(Th, np.ones(3)).flatten(), np.ones(9)).reshape(len(self.xPhys), 9)
        # C_diffs[::3,:]=(C1-C)/h
        # C_diffs[1::3, :] = (C2 - C) /h
        # C_diffs[2::3, :] = (C3 - C) /h



        # for i1 in [1,2,4,6,7,8]:C_diffs[:,i1]/=(4.5)
        # for i1 in [0,3,5]: C_diffs[:, i1] /= (2)

        t1 = time.time() - t1

        t5 = time.time()
        ii = np.array([1, 1, 1, 2, 2, 3, 4, 5, 6]) - 1
        jj = np.array([1, 2, 3, 2, 3, 3, 4, 5, 6]) - 1
        Ce = np.zeros((self.n // 3, 6, 6))
        Ce_diffs = np.zeros((self.n, 6, 6))
        for i in range(9):
            Ce[:, ii[i], jj[i]] = C[:, i]
            Ce[:, jj[i], ii[i]] = C[:, i]
            Ce_diffs[:, ii[i], jj[i]] = C_diffs[:, i]
            Ce_diffs[:, jj[i], ii[i]] = C_diffs[:, i]
        t5 = time.time()-t5

        t6 = time.time()
        Ke = np.zeros((24, 24))
        for i in range(len(self.Bs)):
            Ke = Ke + self.weights[i] * np.matmul(self.BTs[i], np.matmul(Ce, self.Bs[i]))

        Ke_diffs = np.zeros((24, 24))
        for i in range(len(self.Bs)):
            Ke_diffs = Ke_diffs + self.weights[i] * np.matmul(self.BTs[i], np.matmul(Ce_diffs, self.Bs[i]))
        t6 = time.time() - t6

        """thresholding stifness"""
        tmp = np.ones((24, 24))

        pp=self.pp
        penal=(self.rho/0.5)**pp
        sK = Ke.flatten()* (Th[:, np.newaxis, np.newaxis] * tmp).flatten()# * (penal[:, np.newaxis, np.newaxis] * tmp).flatten()
        self.sK=sK
        for iii in range(3):
            tmp = np.ones((24, 24))
            Ke_diffs[iii::3] = (Th[:, np.newaxis, np.newaxis] * tmp).reshape(self.n//3, 24, 24) * Ke_diffs[
                                                                                                          iii::3]

        """solve for obj func"""
        t2=time.time()
        K = coo_matrix((sK, (self.iK, self.jK)), shape=(self.ndof, self.ndof)).tocsc()
        c_style_active_indices = self.free

        f_reduced = np.take(self.f, c_style_active_indices, axis=0)
        K = 0.5 * (K + K.transpose())
        k_reduced = K[self.free, :][:, self.free]
        self.K=k_reduced


        ml = pyamg.smoothed_aggregation_solver(k_reduced.tocsr())
        M = ml.aspreconditioner()
        self.precond=M
        self.u[self.free, 0], info = cg(k_reduced, f_reduced, M=M, atol=1e-10, maxiter=300)
        obj = 1 / 2 * np.dot(self.f.T, self.u)
        t2=time.time()-t2
        """calculationg snsetivites"""
        t3=time.time()
        self.K_diffs=Ke_diffs
        tmp = np.zeros(self.n)

        penal=pp*(self.rho)**(pp-1)
        for i in range(3):
            sK_diff = (-1 / 2 * Ke_diffs[i::3].flatten() * self.u[self.jK].flatten() * self.u[self.iK].flatten()).reshape(self.n // 3, 576)
            # sK_diff = (-1 / 2 * Ke.flatten() * self.u[self.jK].flatten() * self.u[self.iK].flatten()).reshape(self.n // 3, 576)
            tmp[i::3] = sK_diff.sum(1)*penal#

        tmp = np.array(tmp)
        for i in range(3):
            self.grad_obj[i::3] = np.asarray(self.H * (tmp[i::3][np.newaxis].T / self.Hs))[:, 0]*Th
        if grad is not None: grad[:]=self.grad_obj[:]
        t3=time.time()-t3
        print(f"neuro time = {t1}, make tensor = {t5}, integration = {t6}, solve time {t2}, sens time {t3}")
        return obj[0][0]

    @Logger.log_sub('ConstraintDisp')
    def constraintDisp(self,x=None, grad=None):
        """
        function for calculation displacement constraint and its sensitivities
        :param x:
        :param grad:
        :return:
        """
        znak=1

        if len(self.constrainedDofs)==0: return np.array([])

        fx=self.x.copy()
        fx[self.x>0.07]=0.4
        fxPhys=fx.copy()
        for i in range(3):
            fxPhys[i::3] = np.asarray(self.H * fx[i::3][np.newaxis].T / self.Hs)[:, 0]
        C_diffs = self.elasticity_model.predict_sensitivies_CE(fxPhys[0::3], fxPhys[1::3],fxPhys[2::3]) * np.kron(
            np.kron(self.Th, np.ones(3)).flatten(), np.ones(9)).reshape(len(self.xPhys), 9)

        ii = np.array([1, 1, 1, 2, 2, 3, 4, 5, 6]) - 1
        jj = np.array([1, 2, 3, 2, 3, 3, 4, 5, 6]) - 1
        Ce_diffs = np.zeros((self.n, 6, 6))
        for i in range(9):
            Ce_diffs[:, ii[i], jj[i]] = C_diffs[:, i]
            Ce_diffs[:, jj[i], ii[i]] = C_diffs[:, i]

        fKe_diffs = np.zeros((24, 24))
        for i in range(len(self.Bs)):
            fKe_diffs = fKe_diffs + self.weights[i] * np.matmul(self.BTs[i], np.matmul(Ce_diffs, self.Bs[i]))

        for ii in range(3):
            tmp = np.ones((24, 24))
            fKe_diffs[ii::3] = (self.Th[:, np.newaxis, np.newaxis] * tmp).reshape(self.n//3, 24, 24) * fKe_diffs[ii::3]


        f_adj=np.zeros((len(self.constrainedDofs),self.ndof))
        for i in range(len(self.constrainedDofs)):
            f_adj[i,self.constrainedDofs[i]]=1
        u_adj=np.zeros(self.ndof)

        for i in range(len(self.constrainedDofs)):
            u_adj[self.free],info = cg(self.K, f_adj[i,self.free], M=self.precond, atol=1e-10, maxiter=300)

            tmp = np.zeros(self.n)
            for j in range(3):
                sK_diff = (-(self.regc*fKe_diffs[j::3].flatten() + (1-self.regc)*self.K_diffs[j::3].flatten())* u_adj[self.jK].flatten() * self.u[
                    self.iK].flatten()).reshape(self.n // 3, 576)
                tmp[j::3] = sK_diff.sum(1)

            tmp = np.array(tmp)
            for j in range(3):
                self.grad_constr_disp[i,j::3] = znak*np.asarray(self.H * (tmp[j::3][np.newaxis].T / self.Hs))[:, 0] #/self.disp_constr_val[i]

        print(self.u[self.constrainedDofs])
        return znak*(self.u[self.constrainedDofs,0]-self.disp_constr_val) #/self.disp_constr_val

    @Logger.log_end('ConstraintVol')
    def constraintvoluem(self,x=None, grad=None):
        """
        function for calculation volume constraint and its sensitivities
        :param x:
        :param grad:
        :return:
        """
        nell = self.n // 3
        rho_diff = self.elasticity_model.predict_sensitivies_density(self.xPhys[0::3], self.xPhys[1::3], self.xPhys[2::3]).flatten()
        for i in range(3):
            self.grad_constr_vol[i::3] = np.asarray(self.H * (rho_diff[i::3].flatten() / self.Hs.flatten()).T)[:, 0] / nell
        constr = float(self.rho.sum() / nell - self.volfrac)


        if grad is not None: grad[:]=self.grad_constr_vol[:]
        return constr

