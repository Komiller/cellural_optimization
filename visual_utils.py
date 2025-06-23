import numpy as np
import meshio
import pyvista as pv


def dump_new(x, file_name, nelx, nely, nelz, radius_scale=1.0, n_sides=6):
    rods = prepare_data(x, nelx, nely, nelz)
    poly = pv.PolyData()

    points = []
    lines = []
    radii = []

    for i, rod in enumerate(rods):
        start = rod[:3]
        end = rod[3:6]
        radius = rod[6]

        # Добавляем точки
        points.append(start)
        points.append(end)

        # Добавляем линию (2 точки)
        lines.append([2, 2 * i, 2 * i + 1])

        # Радиус для каждой линии
        radii.append(radius * radius_scale)

    # Создаем полидату с линиями
    poly.points = np.array(points)
    poly.lines = np.array(lines)

    # Добавляем данные о радиусах
    poly["Radius"] = np.array(radii)

    # Применяем Tube фильтр
    tubes = poly.tube(radius=0.1, n_sides=n_sides)

    # Сохраняем результат
    tubes.save(f"{file_name}.vtk")

def dump_new2(x, file_name, nelx, nely, nelz, radius_scale=1.0, n_sides=6):
    rods = prepare_data2(x, nelx, nely, nelz)
    poly = pv.PolyData()

    points = []
    lines = []
    radii = []

    for i, rod in enumerate(rods):
        start = rod[:3]
        end = rod[3:6]
        radius = rod[6]

        # Добавляем точки
        points.append(start)
        points.append(end)

        # Добавляем линию (2 точки)
        lines.append([2, 2 * i, 2 * i + 1])

        # Радиус для каждой линии
        radii.append(radius * radius_scale)

    # Создаем полидату с линиями
    poly.points = np.array(points)
    poly.lines = np.array(lines)

    # Добавляем данные о радиусах
    poly["Radius"] = np.array(radii)

    # Применяем Tube фильтр
    tubes = poly.tube(radius=0.1, n_sides=n_sides)

    # Сохраняем результат
    tubes.save(f"{file_name}.vtk")


def write_data_on_hexmesh(file_name,nx,ny,nz,displacement=None,**kwargs):
    dx, dy, dz = 1.0, 1.0, 1.0

    nodes = np.array([[x, y, z] for z in np.arange(0, nz) for y in np.arange(0, ny) for x in np.arange(0, nx)])

    cells = [("hexahedron", np.array([[z * ny * nx + y * nx + x,
                                       z * ny * nx + y * nx + x + 1,
                                       z * ny * nx + (y + 1) * nx + x + 1,
                                       z * ny * nx + (y + 1) * nx + x,
                                       (z + 1) * ny * nx + y * nx + x,
                                       (z + 1) * ny * nx + y * nx + x + 1,
                                       (z + 1) * ny * nx + (y + 1) * nx + x + 1,
                                       (z + 1) * ny * nx + (y + 1) * nx + x]
                                      for z in range(nz - 1) for y in range(ny - 1) for x in range(nx - 1)]))]


    cells = [(name, data.astype(np.int32)) for name, data in cells]
    for key in kwargs:
        kwargs[key]=[kwargs[key]]

    cell_data = kwargs

    # Подготовка point_data (векторное поле перемещений)
    point_data = {}
    if displacement is not None:
        # displacement должен быть массивом формы (n_nodes, 3)
        point_data["Displacement"] = displacement



    # Создаем объект mesh
    mesh = meshio.Mesh(nodes, cells, cell_data=cell_data,point_data=point_data)

    # Записываем сетку в файл формата vtu
    meshio.write(file_name, mesh)


def prepare_data(x,nelx,nely,nelz):
    """
    making PA grid from radii
    :param x: radii

    :param nelx: x length
    :param nely: y length
    :param nelz: z length
    :return: [[coords1,radius1],[coords2,radius2],...]
    """
    #unit cell length
    hx = 1
    hy = 1
    hz = 1
    struts=[]
    for i in range(nelz):
        for j in range(nely):
            for k in range(nelx):
                if x[3 * (i * nelx * nely + j * nelx + k)]>0 :
                    strut1 = [k, j + hy / 2, i + hz / 2, k + hx, j + hy / 2, i + hz / 2,  x[3 * (i * nelx * nely + j * nelx + k)]]
                    struts.append(strut1)

                if x[3 * (i * nelx * nely + j * nelx + k)]+1 > 0:
                    strut2 = [k + hx/2, j, i + hz / 2, k + hx/2, j + hy, i + hz / 2,  x[3 * (i * nelx * nely + j * nelx + k)+1]]
                    struts.append(strut2)
                if x[3 * (i * nelx * nely + j * nelx + k)]+2 > 0:
                    strut3 = [k + hx/2, j + hy / 2, i, k + hx / 2, j + hy / 2, i + hz,  x[3 * (i * nelx * nely + j * nelx + k)+2]]
                    struts.append(strut3)



    return np.array(struts)

def prepare_data2(x,nelx,nely,nelz):
    """
    making PA grid from radii
    :param x: radii

    :param nelx: x length
    :param nely: y length
    :param nelz: z length
    :return: [[coords1,radius1],[coords2,radius2],...]
    """
    #unit cell length
    hx = 1
    hy = 1
    hz = 1
    struts=[]
    for i in range(nelz):
        for j in range(nely):
            for k in range(nelx):
                strut1 = [k+hx/2, j, i, k+hx/2 , j+hy , i+hz ,  x[3 * (i * nelx * nely + j * nelx + k)]]
                strut2 = [k+hx/2, j, i+hz, k+hx/2 , j+hy , i ,  x[3 * (i * nelx * nely + j * nelx + k)]]

                strut3 = [k + hx, j+hy/2, i, k , j + hy/2, i + hz, x[3 * (i * nelx * nely + j * nelx + k)+1]]
                strut4 = [k , j+hy/2, i, k + hx, j + hy / 2, i+hz, x[3 * (i * nelx * nely + j * nelx + k)+1]]

                strut5 = [k , j, i+hz/2, k+hx, j + hy, i + hz / 2, x[3 * (i * nelx * nely + j * nelx + k)+2]]
                strut6 = [k+hx, j , i+hz / 2, k , j + hy, i + hz/2, x[3 * (i * nelx * nely + j * nelx + k)+2]]

                struts.append(strut1)
                struts.append(strut2)
                struts.append(strut3)
                struts.append(strut4)
                struts.append(strut5)
                struts.append(strut6)


    return np.array(struts)

def dump(x,file_name,nelx,nely,nelz):

    rods=prepare_data(x,nelx,nely,nelz)
    # Создаем MultiBlock для хранения всех цилиндров
    multi_block = pv.MultiBlock()

    # Добавляем каждый стержень в MultiBlock
    for rod in rods:
        start = rod[:3]
        end = rod[3:6]
        radius = rod[6]
        cylinder = pv.Cylinder(center=(start + end) / 2, direction=end - start, radius=radius, height=np.linalg.norm(end - start))
        multi_block.append(cylinder)

    # Объединяем все цилиндры в одну сетку
    combined_mesh = multi_block.combine()

    # Сохраняем объединенную сетку в файл
    combined_mesh.save(f"{file_name}.vtk")


def prepare_data_for_bdf(x,nelx,nely,nelz):

    """   making PA grid from radii
    :param x: radii

    :param nelx: x length
    :param nely: y length
    :param nelz: z length
    :return: [[coords1,radius1],[coords2,radius2],...]
    """
    #unit cell length
    hx = 1
    hy = 1
    hz = 1
    struts=[]
    for i in range(nelz):
        for j in range(nely):
            for k in range(nelx):

                strut11 = [2*(k+(2*nely+1)*(j + hy / 2)+(2*nelx+1)*(2*nely+1)*(i + hz / 2)),
                          2*(k+hx/2 + (2*nely+1)*(j + hy / 2) + (2*nelx+1)*(2*nely+1)*(i + hz / 2)),x[3 * (i * nelx * nely + j * nelx + k)]]
                strut12 = [2 * (k+hx/2 + (2 * nely + 1) * (j + hy / 2) + (2 * nelx + 1) * (2 * nely + 1) * (i + hz / 2)),
                          2 * (k + hx + (2 * nely + 1) * (j + hy / 2) + (2 * nelx + 1) * (2 * nely + 1) * (i + hz / 2)),
                          x[3 * (i * nelx * nely + j * nelx + k)]]
                strut21 = [2*(k + hx/2+ (2*nely+1)*j+ (2*nelx+1)*(2*nely+1)*(i + hz / 2)),
                          2*(k + hx/2+ (2*nely+1)*(j + hy/2)+ (2*nelx+1)*(2*nely+1)*(i + hz / 2)),  x[3 * (i * nelx * nely + j * nelx + k)+1]]
                strut22 = [2 * (k + hx / 2 + (2 * nely + 1) * (j + hy/2) + (2 * nelx + 1) * (2 * nely + 1) * (i + hz / 2)),
                          2 * (k + hx / 2 + (2 * nely + 1) * (j + hy) + (2 * nelx + 1) * (2 * nely + 1) * (i + hz / 2)),
                          x[3 * (i * nelx * nely + j * nelx + k) + 1]]
                strut31 = [2*(k + hx/2+ (2*nely+1)*(j + hy / 2)+ (2*nelx+1)*(2*nely+1)*i),
                          2*(k + hx / 2+ (2*nely+1)*(j + hy / 2)+ (2*nelx+1)*(2*nely+1)*(i + hz/2)),  x[3 * (i * nelx * nely + j * nelx + k)+2]]
                strut32 = [2 * (k + hx / 2 + (2 * nely + 1) * (j + hy / 2) + (2 * nelx + 1) * (2 * nely + 1) * (i + hz/2)),
                          2 * (k + hx / 2 + (2 * nely + 1) * (j + hy / 2) + (2 * nelx + 1) * (2 * nely + 1) * (i + hz)),
                          x[3 * (i * nelx * nely + j * nelx + k) + 2]]
                struts.append(strut11)
                struts.append(strut12)
                struts.append(strut21)
                struts.append(strut22)
                struts.append(strut31)
                struts.append(strut32)
    return np.array(struts)

def export_to_nastran(nodes, elements, radii, filename="my_model.bdf"):
    with open(filename, "w") as f:
        f.write("$ NASTRAN input file generated from Python\n")
        f.write("BEGIN BULK\n")

        # Запись узлов
        for i, (x, y, z) in enumerate(nodes):
            f.write(f"GRID,{i + 1},,{x:.6f},{y:.6f},{z:.6f}\n")

        # Запись стержней (CBAR или CROD)
        for i, (n1, n2) in enumerate(elements):
            pid = i + 1  # ID свойства элемента
            mid = 1      # ID материала (можно задать отдельно)
            # Запись свойства стержня (PID)
            """
            1       2   3   4   5   6   7   8   9
            pbar    pid MID A   I1  I2  J   NSM
                    C1  C2  D1  D2  E1  E2  F1  F2
                    K1  K2  I12
            """
            A=np.pi*radii[i]**2
            I1=np.pi*radii[i]**4/4
            J=2*I1
            K1=1.27*0.885*A
            f.write(f'$ Femap PropShape {pid} : 5,0,{radii[i]:.6f},0.,0.,0.,0.,0.  \n')
            f.write(f'$ Femap PropMethod {pid} : 5,0,1,0.\n')
            f.write(f'$ Femap PropOrient {pid} : 5,0,0.,1.,2.,3.,4.,0.,0.,{-radii[i]:.6f} \n')
            f.write(f"PBAR,{pid},{mid},{A:.6f},{I1:.6f},{I1:.6f},{J:.6f},   \n")
            f.write(f",{0:.6f},{-radii[i]:.6f},{radii[i]:.6f},{0:.6f},{0:.6f},{radii[i]:.6f},{-radii[i]:.6f},{0:.6f}   \n")
            f.write(f",{K1:.6f},{K1:.6f},,,,,   \n")
            # Запись элемента (CBAR)
            f.write(f"CBAR,{i + 1},{pid},{n1 + 1},{n2 + 1},{1:.6f},{1:.6f},{1:.6f},\n")

        f.write("ENDDATA\n")


def write_to_bdf(x,nelx,nely,nelz,fname):


    nodes=[]
    all_nodes_to_active=[]
    count=0
    for i in range(2*nelz+1):
        for j in range(2*nely+1):
            for k in range(2*nelx+1):
                if (k%2==0 and i%2==1 and j%2==1) or (k%2==1 and i%2==0 and j%2==1) or (k%2==1 and i%2==1 and j%2==0) or (k%2==1 and i%2==1 and j%2==1):
                    nodes.append([0.5*k,0.5*j,0.5*i])
                    all_nodes_to_active.append(count)
                    count+=1
                else:
                    all_nodes_to_active.append(0)


    nodes=np.array(nodes)
    all_nodes_to_active=np.array(all_nodes_to_active)
    rods=prepare_data_for_bdf(x,nelx,nely,nelz)
    elements=rods[:,:2].astype(np.int32)
    elements[:,0]=all_nodes_to_active[elements[:,0]]
    elements[:,1]=all_nodes_to_active[elements[:,1]]

    # Пример данных: радиусы стержней
    radii = rods[:,-1]

    export_to_nastran(nodes, elements, radii,fname+'.bdf')

if __name__ == '__main__':
    nelx,nely,nelz=1,1,1
    dump(np.ones(3*nelx*nely*nelz)*0.1,'PA1_PUC.vtu',nelx,nely,nelz)