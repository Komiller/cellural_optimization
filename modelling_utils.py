import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt


def plot_disp(a,fixed,nelx,nely,nelz):
    a=a/np.abs(a).max()
    points = []
    for k in range((nelz + 1)):
        for j in range((nely + 1)):
            for i in range((nelx + 1)):
                points.append([i, j, k])
    # Пример данных: массив точек и массив перемещений
    points = np.array(points)  # Координаты точек в 3D

    displacements = a.reshape(len(points), 3)  # Перемещения точек

    # Разделяем координаты точек на x, y, z
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # fixed_node=fixed//3
    # fixed_dof=fixed%3
    # x_f=points[fixed_node,0]
    # y_f = points[fixed_node, 1]
    # z_f = points[fixed_node, 2]

    # Разделяем перемещения на dx, dy, dz
    nn=1 #u.max()
    dx = displacements[:, 0]/nn
    dy = displacements[:, 1]/nn
    dz = displacements[:, 2]/nn
    # Создаем 3D график
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Рисуем стрелки перемещений
    ax.quiver(x, y, z, dx, dy, dz, color='red', length=1, arrow_length_ratio=0.3)
    # ax.scatter(x_f,y_f,z_f, c='blue', s=30, label='Узлы')
    # for i,dof in enumerate(fixed_dof):
    #     # Смещаем текст, чтобы не накладывался на точку
    #     ax.text(x_f[i] + 0.05, y_f[i] + 0.05, z_f[i] - 0.8*dof, str(dof), color='red', fontsize=12)

    # Настройка графика
    ax.set_xlim([min(x) - 1, max(x) + 1])
    ax.set_ylim([min(y) - 1, max(y) + 1])
    ax.set_zlim([390,400])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Визуализация перемещений точек в 3D')

    # Показываем график
    plt.show()



coeffs = [0.2969,-0.1269,-0.3516,0.2843,-0.1015]
acc=0.01
t=0.12
def profile_0012(center,nelx,nely,cord,gamma=0):
    x=(center[0] - center[2]*np.tan(gamma/180*np.pi) )/cord
    y=(center[1]-nely//2)/cord
    if x<0 or x>1:return False
    y_t=(t/0.2)*(coeffs[0]*np.sqrt(x)+coeffs[1]*x+coeffs[2]*x*x+coeffs[3]*x*x*x+coeffs[4]*x*x*x*x)
    if (y<=y_t+acc) and (y>=-acc-y_t):return True
    return False

def profile_cyllinder(center,nelx,nely):
    x=center[0]
    y=center[1]

    if ((x-nelx//2)**2 + (y-nely//2)**2)<=(nelx//2)**2:return True
    return False

def f_c(nelx,nely):
    f=np.zeros(3*(nelx+1)*(nely+1))
    for x in range(nelx+1):
        for y in range(nely+1):
            if ((x - nelx // 2) ** 2 + (y - nely // 2) ** 2) <= (nelx // 2) ** 2 and not(x==y==nelx//2):
                id=x+(nelx+1)*y
                f[3*id]=-(y - nely // 2)/((x - nelx // 2) ** 2 + (y - nely // 2) ** 2)**0.5
                f[3 * id+1] = (x - nelx // 2) / ((x - nelx // 2) ** 2 + (y - nely // 2) ** 2) ** 0.5
    return f

element_size=1
half_size = element_size / 2
vertices_offsets = [
    [-half_size, -half_size, -half_size],
    [half_size, -half_size, -half_size],
    [half_size, half_size, -half_size],
    [-half_size, half_size, -half_size],
    [-half_size, -half_size, half_size],
    [half_size, -half_size, half_size],
    [half_size, half_size, half_size],
    [-half_size, half_size, half_size]
]

# Определение граней куба (индексы вершин)
faces = [
    [0, 1, 2, 3],  # нижняя грань
    [4, 5, 6, 7],  # верхняя грань
    [0, 1, 5, 4],  # передняя грань
    [2, 3, 7, 6],  # задняя грань
    [0, 3, 7, 4],  # левая грань
    [1, 2, 6, 5]  # правая грань
]

def outer_nodes(nelx,nely,nelz,xmask):
    face_dict = defaultdict(int)
    for x in range(nelx):
        for y in range(nely):
            for z in range(nelz):

                idx=x+nelx*y+nelx*nely*z
                if not xmask[idx]:continue
                cx,cy,cz=x+half_size,y+half_size,z+half_size

                for face in faces:
                    # Получаем координаты вершин грани
                    face_vertices = []
                    for vertex_idx in face:
                        offset = vertices_offsets[vertex_idx]
                        vertex = (cx + offset[0], cy + offset[1], cz + offset[2])
                        # Округляем координаты для избежания ошибок из-за погрешности float
                        rounded_vertex = tuple(round(coord, 6) for coord in vertex)
                        face_vertices.append(rounded_vertex)
                        face_key = tuple(sorted(face_vertices))
                    face_dict[face_key] += 1
                    if idx in [1,5]:print(face_key)

                print('_________')



    surface_faces = [face for face, count in face_dict.items() if count == 1]
    surface_nodes = set()
    for face in surface_faces:
        for vertex in face:
            surface_nodes.add(vertex)

    inner_faces = [face for face, count in face_dict.items() if count > 2]
    inner_nodes = set()
    for face in inner_faces:
        for vertex in face:
            inner_nodes.add(vertex)

    return np.array(list(surface_nodes-inner_nodes))

def apply_presure_load_on_cyllinder(nodes,nelx,nely,nelz):
    f=np.zeros(3*(nelx+1)*(nely+1)*(nelz+1))
    for node in nodes:
        nx,ny,nz=node
        id=int(nx+(nelx+1)*ny+(nelx+1)*(nely+1)*nz)
        dx  = nx-nelx//2
        dy = ny-nely//2
        if dx==dy==0:continue
        if nz==0 or nz == nelz: continue
        if dy<0:continue
        f[3*id] = dx/(dx**2+dy**2)**0.5
        f[3 * id+1] = dy / (dx ** 2 + dy ** 2) ** 0.5
    return f

def apply_presure_load_on_prof0012(nodes,nelx,nely,nelz,cord0,gamma):
    print(len(nodes))
    f=np.zeros(3*(nelx+1)*(nely+1)*(nelz+1))
    for node in nodes:
        cord = cord0 - (cord0 - 20) * node[2] / nelz
        nx=max((node[0]-node[2]*np.tan(gamma/180*np.pi))/cord,0)
        ny=(node[1]-nely//2)/nely
        nz=node[2]

        dx  = node[0]-cord//2
        dy = node[1]-nely//2

        if node[2]==nelz:continue
        if dx==dy==0:continue
        if nz==0 or nz == nelz: continue
        if ny<0:continue
        if nx==0:continue
        y_t=(t/0.2)*(coeffs[0]*np.sqrt(nx)+coeffs[1]*nx+coeffs[2]*nx*nx+coeffs[3]*nx*nx*nx+coeffs[4]*nx*nx*nx*nx)
        dy_t=(t/0.2)*(coeffs[0]*(1/2)/np.sqrt(nx)+coeffs[1]+coeffs[2]*2*nx+coeffs[3]*3*nx*nx+coeffs[4]*4*nx*nx*nx)
        dy_t2=(t/0.2)*(coeffs[0]*(-1/2)*(1/2)/nx**(-3/2)+coeffs[2]*2+coeffs[3]*3*2*nx+coeffs[4]*4*3*nx*nx)
        if nx<=1/4:
            magnitude=nx*4*0.01
        else: magnitude = 4/3*(1-nx)*0.01
        norm=np.sqrt(dy_t**2+1)
        id=int(node[0]+(nelx+1)*node[1]+(nelx+1)*(nely+1)*node[2])
        f[3*id] = dy_t/norm*magnitude
        f[3 * id+1] = np.sign(dy_t2) / norm*magnitude
    print(f.sum())
    return f

if __name__ == '__main__':
    nelx,nely,nelz=10,10,2
    x = np.zeros(3*nelx*nely*nelz)

    xmask = np.zeros(nelx * nely * nelz, dtype=bool)
    for xx in range(nelx):
        for yy in range(nely):
            for zz in range(nelz):
                center = (xx + 0.5, yy + 0.5)
                idx = xx + nelx * yy + nelx * nely * zz
                xmask[idx] = profile_cyllinder(center, nelx, nely)

    outer_nodes_arr=outer_nodes(nelx,nely,nelz,xmask)
    f=apply_presure_load_on_cyllinder(outer_nodes_arr,nelx,nely,nelz)
    plot_disp(f,np.array([0,1,2]),nelx,nely,nelz)


    print(xmask.reshape((nelx, nely)).T)
