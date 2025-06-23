import pythoncom
import PyFemap
import sys
import win32com.client.gencache
import numpy as np
#from model import model

def femap_init():
    try:
        win32com.client.gencache.is_readonly=False
        existObj = pythoncom.connect(PyFemap.model.CLSID) #Grabs active model
        app = PyFemap.model(existObj)#Initializes object to active mode
        return app
    except:
        sys.exit("femap not open") #Exits program if there is no active femap model



def apply_2d_PBC():
    app=femap_init()
    nodes=app.feNode
    out=nodes.GetAllArray(0)
    hn={} #hashed nodes
    count=0
    coords=np.round(np.array(out[3],dtype=np.float64),3)
    for id in out[2]:
        hash_=hash(tuple(coords[3*count:3*count+3]))
        hn[hash_]=id
        count=-~count

    a=0.5
    b=0.5
    n3=hash((-0.5,0.5,0.5))
    bceq=app.feBCEqn

    import time
    t=time.time()


    def PBC(x1,x2,x3):
        u1 = {'d1':x1,'d2':0,'d3':0.5*x2,'d4':x1*x3,'d5':0,'d6':0.5*x2*x3}
        u2 = {'d1':0 , 'd2': x1, 'd3': 0.5 * x2, 'd4': 0, 'd5': x1 * x3, 'd6': 0.5 * x2 * x3}
        u3 = {'d1': 0, 'd2': 0, 'd3': 0, 'd4': -0.5*x1**2, 'd5': -0.5*x2**2, 'd6': -0.25 * x1 * x2}
        return np.array([[u1['d1'],u1['d2'],u1['d3'],u1['d4'],u1['d5'],u1['d6']],
                         [u1['d1'],u2['d2'],u2['d3'],u2['d4'],u2['d5'],u2['d6']],
                         [u2['d1'],u3['d2'],u3['d3'],u3['d4'],u3['d5'],u3['d6']]])


    def put_constr(pbc,constr_n,n1_id,n2_id):
        global bceq
        if n1_id is not None and n2_id is not None:
            for ui in range(3):
                mask = pbc[ui] != 0

                pbc_t = pbc[ui,mask]
                dId_t=dId[mask]
                ids = [n2_id, n1_id]
                dof = [ui+1, ui+1]
                equation= [-1,1]
                for num in range(len(dId_t)):
                    ids.append(hn[n3])
                    dof.append(dId_t[num])
                    equation.append(pbc_t[num])
                    bceq.PutAll(constr_n+ui, 2, constr_n+ui, len(ids), ids, dof,equation, 8312, 1)
            constr_n+=3
        return constr_n


    constr_n=1
    dId=np.array([1,2,3,4,5,6])
    dH=0.02
    Lx=int(1/dH)



    #face2 to face3
    for i in range(1,Lx):
        for j in range(0,Lx+1):
            y=round(dH * i,3)
            z=round(dH * j,3)
            n1 = hash((0, y, z))
            n2 = hash((1, y, z))
            n1_id=hn.get(n1)
            n2_id=hn.get(n2)
            if n1_id is not None and n2_id is not None:
                bceq.PutAll(constr_n, 2, constr_n, 4, [hn[n1], hn[n2], hn[n3], hn[n3]], [1, 1, 1, 4], [1, -1, -2*a,-2*a*z], 8312, 1) #u1
                bceq.PutAll(constr_n+1, 2, constr_n+1, 4, [hn[n1], hn[n2], hn[n3], hn[n3]], [2, 2, 3, 6 ], [1, -1, -a,-a*z], 8312, 1) #u2
                bceq.PutAll(constr_n+2, 2, constr_n+2, 3, [hn[n1], hn[n2], hn[n3]], [3, 3, 6], [1, -1, a*y], 8312, 1) #u3
                constr_n+=3


    #face2 to face3
    for i in range(1,Lx):
        for j in range(0,Lx+1):
            x=round(dH * i,3)
            z=round(dH * j,3)
            n1 = hash((x, 0, z))
            n2 = hash((x, 1, z))
            n1_id=hn.get(n1)
            n2_id=hn.get(n2)
            if n1_id is not None and n2_id is not None:
                bceq.PutAll(constr_n, 2, constr_n, 4, [hn[n1], hn[n2], hn[n3], hn[n3]], [1, 1, 3, 6], [1, -1, -b,-b*z], 8312, 1) #u1
                bceq.PutAll(constr_n+1, 2, constr_n+1, 4, [hn[n1], hn[n2], hn[n3], hn[n3]], [2, 2, 2, 5 ], [1, -1, -2*b,-2*b*z], 8312, 1) #u2
                bceq.PutAll(constr_n+2, 2, constr_n+2, 3, [hn[n1], hn[n2], hn[n3]], [3, 3, 6], [1, -1, b*x], 8312, 1) #u3
                constr_n+=3

    #edge1 to edge3
    for i in range(0,Lx+1):
            z=round(dH * i,3)
            n1 = hash((0, 0, z))
            n2 = hash((1, 1, z))
            n1_id=hn.get(n1)
            n2_id=hn.get(n2)
            if n1_id is not None and n2_id is not None:
                bceq.PutAll(constr_n, 2, constr_n, 6, [hn[n2], hn[n1], hn[n3], hn[n3], hn[n3], hn[n3]], [1, 1, 1, 3, 4, 6], [-1, 1, -2*a,-b,-2*a*z,-b*z], 8196, 1) #u1
                bceq.PutAll(constr_n+1, 2, constr_n+1, 6, [hn[n2], hn[n1], hn[n3], hn[n3], hn[n3], hn[n3]], [2, 2, 3, 2, 6, 5], [-1, 1, -a,-2*b,-a*z,2*b*z], 8196, 1) #u2
                bceq.PutAll(constr_n+2, 2, constr_n+2, 2, [hn[n2], hn[n1]], [3, 3], [-1, 1], 8196, 1) #u3
                constr_n+=3

    #edge1 to edge2
    for i in range(0,Lx+1):
            z=round(dH * i,3)
            n1 = hash((0, 0, z))
            n2 = hash((1, 0, z))
            n1_id=hn.get(n1)
            n2_id=hn.get(n2)
            if n1_id is not None and n2_id is not None:
                bceq.PutAll(constr_n, 2, constr_n, 4, [hn[n2], hn[n1], hn[n3], hn[n3]], [1, 1, 1, 4],[-1, 1, -2 * a, -2 * a * z], 8196, 1)  # u1
                bceq.PutAll(constr_n + 1, 2, constr_n + 1, 4, [hn[n2], hn[n1], hn[n3], hn[n3]], [2, 2, 3, 6],[-1, 1, -a, -a * z], 8196, 1)  # u2
                bceq.PutAll(constr_n + 2, 2, constr_n + 2, 3, [hn[n2], hn[n1], hn[n3]], [3, 3, 6], [-1, 1, -a * b], 8196,1)  # u3
                constr_n+=3

    #edge1 to edge4
    for i in range(0,Lx+1):
            z=round(dH*i,3)
            n1 = hash((0, 0, z))
            n2 = hash((0, 1, z))
            n1_id=hn.get(n1)
            n2_id=hn.get(n2)
            if n1_id is not None and n2_id is not None:
                bceq.PutAll(constr_n, 2, constr_n, 4, [hn[n2], hn[n1], hn[n3], hn[n3]], [1, 1, 3, 6],[-1, 1, -b, -b * z], 8196, 1)  # u1
                bceq.PutAll(constr_n + 1, 2, constr_n + 1, 4, [hn[n2], hn[n1], hn[n3], hn[n3]], [2, 2, 2, 5],[-1, 1, -2 * b, -2 * b * z], 8196, 1)  # u2
                bceq.PutAll(constr_n + 2, 2, constr_n + 2, 3, [hn[n2], hn[n1], hn[n3]], [3, 3, 6], [-1, 1, -a * b], 8196,1)  # u3
                constr_n+=3

def apply_3d_PBC():
    global constr_n
    app = femap_init()
    nodes=app.feNode
    out=nodes.GetAllArray(0)
    hn={} #hashed nodes
    count=0
    coords=np.round(np.array(out[3],dtype=np.float64),3)
    for id in out[2]:
        hash_=hash(tuple(coords[3*count:3*count+3]))
        hn[hash_]=id
        count=-~count
    global b
    global h
    global t
    b=1 #x length
    h=1 #y length
    t=1 #z length

    n3=hash((-0.5,0.5,0.5))

    bceq=app.feBCEqn



    constr_n=1
    dId=np.array([1,2,3,4,5,6])
    dH=0.02
    Lx=int(1/dH)


    def put_constr(n1,n2,i,j,k):
        global constr_n
        global b
        global h
        global t
        du = np.array([-i * b, -j * h, -k * t])
        dv = np.array([-j * h, -k * t])
        dw = np.array([-k * t])

        idu = np.array([1, 6, 5])
        idv = np.array([2, 4])
        idw = np.array([3])

        masku = (du != 0)
        maskv = (dv != 0)
        maskw = (dw != 0)

        bceq.PutAll(constr_n, 2, constr_n, 2 + masku.sum(), [hn[n1], hn[n2]] + [hn[n3]] * masku.sum(),
                    [1, 1] + list(idu[masku]), [1, -1] + list(du[masku]), 8312, 1)  # u1

        bceq.PutAll(constr_n + 2, 2, constr_n + 2, 2 + maskw.sum(), [hn[n1], hn[n2]] + [hn[n3]] * maskw.sum(),
                    [3, 3] + list(idw[maskw]), [1, -1] + list(dw[maskw]), 8312, 1)  # u3

        bceq.PutAll(constr_n + 1, 2, constr_n + 1, 2 + maskv.sum(), [hn[n1], hn[n2]] + [hn[n3]] * maskv.sum(),
                    [2, 2] + list(idv[maskv]), [1, -1] + list(dv[maskv]), 8312, 1)  # u2
        constr_n += 3

    #face right to face left
    for i in range(1,Lx):
        for j in range(1,Lx):
            y=round(dH * i,3)
            z=round(dH * j,3)
            n1 = hash((1, y, z))
            n2 = hash((0, y, z))
            n1_id=hn.get(n1)
            n2_id=hn.get(n2)

            if n1_id is not None and n2_id is not None:put_constr(n1,n2,1,0,0)

    #face upper to face lower
    for i in range(1,Lx):
        for j in range(1,Lx):
            x=round(dH * i,3)
            z=round(dH * j,3)
            n1 = hash((x, 1, z))
            n2 = hash((x, 0, z))
            n1_id=hn.get(n1)
            n2_id=hn.get(n2)
            if n1_id is not None and n2_id is not None:put_constr(n1,n2,0,1,0)

    #face front to face back
    for i in range(1,Lx):
        for j in range(1,Lx):
            x=round(dH * i,3)
            y=round(dH * j,3)
            n1 = hash((x, y, 1))
            n2 = hash((x, y, 0))
            n1_id=hn.get(n1)
            n2_id=hn.get(n2)
            if n1_id is not None and n2_id is not None:put_constr(n1,n2,0,0,1)



    """edges"""

    #edge right back to edge left back
    for i in range(1,Lx):
            y=round(dH * i,3)
            n1 = hash((1, y, 0))
            n2 = hash((0, y, 0))
            n1_id=hn.get(n1)
            n2_id=hn.get(n2)
            if n1_id is not None and n2_id is not None:put_constr(n1,n2,1,0,0)

    #edge right front to edge left front
    for i in range(1,Lx):
            y=round(dH * i,3)
            n1 = hash((1, y, 1))
            n2 = hash((0, y, 1))
            n1_id=hn.get(n1)
            n2_id=hn.get(n2)
            if n1_id is not None and n2_id is not None:put_constr(n1,n2,1,0,0)

    #edge upper back to edge lower back
    for i in range(1,Lx):
            x=round(dH * i,3)
            n1 = hash((x, 1, 0))
            n2 = hash((x, 0, 0))
            n1_id=hn.get(n1)
            n2_id=hn.get(n2)
            if n1_id is not None and n2_id is not None:put_constr(n1,n2,0,1,0)

    #edge upper front to edge lower front
    for i in range(1,Lx):
            x=round(dH * i,3)
            n1 = hash((x, 1, 1))
            n2 = hash((x, 0, 1))
            n1_id=hn.get(n1)
            n2_id=hn.get(n2)
            if n1_id is not None and n2_id is not None:put_constr(n1,n2,0,1,0)

    #edge left front to edge left back
    for i in range(1,Lx):
            y=round(dH * i,3)
            n1 = hash((0, y, 1))
            n2 = hash((0, y, 0))
            n1_id=hn.get(n1)
            n2_id=hn.get(n2)
            if n1_id is not None and n2_id is not None:put_constr(n1,n2,0,0,1)

    #edge lower front to edge lower back
    for i in range(1,Lx):
            x=round(dH * i,3)
            n1 = hash((x, 0, 1))
            n2 = hash((x, 0, 0))
            n1_id=hn.get(n1)
            n2_id=hn.get(n2)
            if n1_id is not None and n2_id is not None:put_constr(n1,n2,0,0,1)

    #edge lower right to edge lower left
    for i in range(1,Lx):
            z=round(dH * i,3)
            n1 = hash((1, 0, z))
            n2 = hash((0, 0, z))
            n1_id=hn.get(n1)
            n2_id=hn.get(n2)
            if n1_id is not None and n2_id is not None:put_constr(n1,n2,1,0,0)

    #edge upper left to edge lower left
    for i in range(1,Lx):
            z=round(dH * i,3)
            n1 = hash((0, 1, z))
            n2 = hash((0, 0, z))
            n1_id=hn.get(n1)
            n2_id=hn.get(n2)
            if n1_id is not None and n2_id is not None:put_constr(n1,n2,0,1,0)

    #edge upper right to edge lower right
    for i in range(1,Lx):
            z=round(dH * i,3)
            n1 = hash((1, 1, z))
            n2 = hash((1, 0, z))
            n1_id=hn.get(n1)
            n2_id=hn.get(n2)
            if n1_id is not None and n2_id is not None:put_constr(n1,n2,0,1,0)




    for i in range(2):
        for j in range(2):
            for k in range(2):
                if (i,j,k)==(0,0,0): continue
                n1 = hash((i,j,k))
                n2 = hash((0, 0, 0))


                n1_id = hn.get(n1)
                n2_id = hn.get(n2)
                if n1_id is not None and n2_id is not None: put_constr(n1,n2,i,j,k)


if __name__ == '__main__':
    apply_3d_PBC()
