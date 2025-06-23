import numpy as np
from scipy.sparse import coo_matrix
from preprocessor import generate_voxel
from model import model
import pythoncom
import PyFemap
import sys
import win32com.client.gencache
from femap_integration import create_matirial_iso,create_property

def voxelize(file):
    try:
        win32com.client.gencache.is_readonly=False
        existObj = pythoncom.connect(PyFemap.model.CLSID) #Grabs active model
        app = PyFemap.model(existObj)#Initializes object to active mode
    except:
        sys.exit("femap not open") #Exits program if there is no active femap model

    nelx, nely, nelz = 8,8,8
    #x = np.load('C:/Users/mille/Documents/cellural_optimiation/optimization_results/opt_res_2025-04-25_20-00-02/radii_100.npy').flatten()
    x=np.ones(nelx*nely*nelz*3)*0.2
    elasticity_model=model('weights_box','scalers_box')

    
    print(nelx,nely,nelz)
    x=x.flatten()
    rmin=1.5

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


    xPhys=x.copy()
    """applying filter"""
    for i in range(3):
        xPhys[i::3]= np.asarray(H*x[i::3][np.newaxis].T/Hs)[:,0]


    rho=elasticity_model.predict_value_density(xPhys[0::3],xPhys[1::3],xPhys[2::3])
    a=1e4
    b=0.08
    Th= list(map(lambda zu: 1/2*(1+np.tanh(a*(zu-b))),rho))
    xPhys=xPhys*np.kron(Th,np.ones(3)).flatten()

    nodes={}
    elements=[]
    nvoxel=10
    h_voxel=1/nvoxel
    node_id=1
    dens=0
    for x in range(nelx):
        for y in range(nely):
            for z in range(nelz):
                uc_n=x+y*nelx+z*nelx*nely
                r=[0,0,0]
                if x==z:r=[0.2,0.2,0.2]
                if (x + 1) == z or x == (z + 1): r = [0.2, 0.2, 0.2]
                #voxel, density, lambdas, mus, thermals=generate_voxel(nvoxel,'PA.txt',[r[0]]*4+[r[1]]*4+[r[2]]*4)
                voxel, density, lambdas, mus, thermals = generate_voxel(nvoxel, 'PA.txt',[xPhys[3*uc_n]]*1+[xPhys[3*uc_n+1]]*1+[xPhys[3*uc_n+2]]*1)


                for vx in range(nvoxel):
                    for vy in range(nvoxel):
                        for vz in range(nvoxel):
                            if voxel[vz,vx,vy]==0: continue
                            dens+=1
                            elm_nodes=np.array([(x+vx*h_voxel,y+(vy+1)*h_voxel,z+vz*h_voxel),(x+vx*h_voxel,y+(vy+1)*h_voxel,z+(vz+1)*h_voxel),
                                       (x+vx*h_voxel,y+vy*h_voxel,z+(vz+1)*h_voxel),(x+vx*h_voxel,y+vy*h_voxel,z+vz*h_voxel),
                                       (x + (vx+1) * h_voxel, y + (vy + 1) * h_voxel, z + vz * h_voxel),(x + (vx+1) * h_voxel, y + (vy + 1) * h_voxel, z + (vz + 1) * h_voxel),
                                       (x + (vx+1) * h_voxel, y + vy * h_voxel, z + (vz + 1) * h_voxel),(x + (vx+1) * h_voxel, y + vy * h_voxel, z + vz * h_voxel)])
                            elm_nodes=np.round(elm_nodes,4)
                            for node in elm_nodes:
                                nid=nodes.get(tuple(node))
                                if nid is None:
                                    nodes[tuple(node)]=node_id
                                    nid = node_id
                                    node_id+=1
                                elements.append(nid)
                            for _ in range(12): elements.append(0)
                print(node_id)

    xyz=[]
    for key in nodes:
        xyz.append(key[0])
        xyz.append(key[1])
        xyz.append(key[2])
    ids=np.arange(1,node_id)
    node=app.feNode
    node.PutCoordArray(len(ids),ids,xyz)

    elem=app.feElem
    nel=len(elements)//20
    print(elem.PutAllArray(nel,np.arange(1,nel+1),[1]*nel,[25]*nel,[8]*nel,[1]*nel,[124]*nel,[0]*2*nel,[0]*3*nel,[0]*nel*6,[0]*nel*12,[0]*nel,[0]*nel,elements,[0]*2*nel,[0]*2*nel))
    print(dens/nelx/nely/nelz/nvoxel**3)
if __name__ == '__main__':
    fff=np.ones(2*2*2*3)*0.2
    create_matirial_iso(200, 0.3)
    create_property()
    voxelize(fff)
