import numpy as np
import matplotlib.pyplot as plt
import re
from visual_utils import write_data_on_hexmesh


"""degrees of freedom mask intialisation"""
# for xx in range(nelx + 1):
#     for yy in range(nely + 1):
#         for zz in range(nelz + 1):
#             idx = xx + (nelx + 1) * yy + (nelx + 1) * (nely + 1) * zz
#             adj_elem = []
#             if xx < nelx and yy < nely and zz < nelz:
#                 adj_elem.append(xx + (nelx) * yy + (nelx) * (nely) * zz)
#
#             if xx < nelx and yy < nely and zz > 0:
#                 adj_elem.append(xx + (nelx) * yy + (nelx) * (nely) * (zz - 1))
#             if xx < nelx and yy > 0 and zz < nelz:
#                 adj_elem.append(xx + (nelx) * (yy - 1) + (nelx) * (nely) * zz)
#             if xx > 0 and yy < nely and zz < nelz:
#                 adj_elem.append((xx - 1) + (nelx) * (yy) + (nelx) * (nely) * zz)
#
#             if xx < nelx and yy > 0 and zz > 0:
#                 adj_elem.append(xx + (nelx) * (yy - 1) + (nelx) * (nely) * (zz - 1))
#             if xx > 0 and yy < nely and zz > 0:
#                 adj_elem.append((xx - 1) + (nelx) * yy + (nelx) * (nely) * (zz - 1))
#             if xx > 0 and yy > 0 and zz < nelz:
#                 adj_elem.append((xx - 1) + (nelx) * (yy - 1) + (nelx) * (nely) * zz)
#
#             if xx > 0 and yy > 0 and zz > 0:
#                 adj_elem.append((xx - 1) + (nelx) * (yy - 1) + (nelx) * (nely) * (zz - 1))
#             number = bool(xmask[adj_elem].sum())
#             dof_mask[3 * idx] = number
#             dof_mask[3 * idx + 1] = number
#             dof_mask[3 * idx + 2] = number

# dof_mask=np.load('wing_dof_mask.npy')
# # np.save('wing_dof_mask.npy',dof_mask)
# disp[dof_mask] = disp1.flatten()
#
# u=[]
# for zz in range(400):
#     id1=(nelx+1)*(nely+1)*zz
#     id2=nelx+(nelx+1)*nely+(nelx+1)*(nely+1)*zz
#     u_slice=disp[3*id1+1:(3*id2+3):3]
#     u_slice=u_slice[u_slice!=0]
#
#     u.append(u_slice.sum()/len(u_slice))
#     if u[-1]<-100:u[-1]=u[-2]
#     if u[-1] > 1000 : u[-1] = u[-2]
#
# fig, ax = plt.subplots()
# ax.plot(np.arange(len(u)),u)
# ax.set_xlim([0,nelz])
# print(len(u))
#
# plt.show()
#
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
#             elm_disp[idx]=min(eld.sum()/8,200)
# write_data_on_hexmesh('wing_disps.vtu',nelx+1,nely+1,nelz+1,Th=xmask.astype(np.int32),eld=elm_disp)
# import matplotlib.pyplot as plt
# import numpy as np
#
# # Данные: 4 пары значений
# pairs = [
#     ('Растяжение', (1, 278/215)),
#     ('Изгиб', (1, 11915/11758)),
#     ('Давление', (1, 122504/100584)),
#     ('Кручение', (1, 651/1070))
# ]
#
# # Разделяем названия пар и значения
# labels = [pair[0] for pair in pairs]
# values_left = [pair[1][0] for pair in pairs]  # Левые значения (синие)
# values_right = [pair[1][1] for pair in pairs]  # Правые значения (красные)
#
# # Позиции для столбцов
# x = np.arange(len(labels)) * 2  # Умножаем на 2 для создания пропуска между парами
# width = 0.35  # Ширина столбцов
#
# # Создаем диаграмму
# fig, ax = plt.subplots(figsize=(10, 6))
#
# # Левые столбцы (синие)
# rects1 = ax.bar(x - width/2, values_left, width, label='Топология 1', color=(29/255,73/255,186/255))
# # Правые столбцы (красные)
# rects2 = ax.bar(x + width/2, values_right, width, label='Топология 2', color=(175/255,24/255,34/255))
#
# # Настройки осей и подписей
# ax.set_ylabel('Значения')
# ax.set_title('Столбчатая диаграмма с парами значений')
# ax.set_xticks(x)
# ax.set_xticklabels(labels)
# ax.legend()
#
# # Добавляем подписи значений над столбцами
# def autolabel(rects):
#     for rect in rects:
#         height = rect.get_height()
#         ax.annotate('{}'.format(height),
#                     xy=(rect.get_x() + rect.get_width() / 2, height),
#                     xytext=(0, 3),  # 3 points vertical offset
#                     textcoords="offset points",
#                     ha='center', va='bottom')
#
#
# # Настраиваем отступы между группами столбцов
# plt.margins(x=0.1)
# plt.tight_layout()
#
# plt.show()

content_pattern = re.compile(r'Obj\s*=\s*(-?\d+\.\d{3})')
fig, ax = plt.subplots()
files=['opt_res_2025-06-17_01-05-54']
content_pattern = re.compile(r'ConstraintVol\s*=\s*(-?\d+\.\d{3})')
for fname in files:
    with open(f"optimization_results/{fname}/log.txt",'r') as f:
        obnums = []
        ob_count=0
        for line in f:
            content_match = content_pattern.search(line)
            if content_match:
                obnum = (float(content_match.group(1)))
                #if ob_count == 100: continue
                obnums.append(obnum)
                ob_count+=1
        obnums=np.array(obnums)/obnums[0]
        ax.plot(np.arange(len(obnums)), obnums,label='Ограничение')
print(len(obnums))
content_pattern = re.compile(r'Obj\s*=\s*(-?\d+\.\d{3})')
for fname in files:
    with open(f"optimization_results/{fname}/log.txt",'r') as f:
        obnums = []
        ob_count=0
        for line in f:
            content_match = content_pattern.search(line)
            if content_match:
                obnum = float(content_match.group(1))
                #if ob_count == 100: continue
                obnums.append(obnum)
                ob_count+=1
        obnums = np.array(obnums) / obnums[0]
        ax.plot(np.arange(len(obnums)), obnums,label='Целевая функция')

# ax.set_ylim([0,100])
ax.set_xlim([0,200])
plt.grid(True)
plt.legend()
plt.show()




# """разные алгоритмы"""
# hsv_colors = [(0.64,0.7,0.8)]*3
# rgb_colors = [mcolors.hsv_to_rgb(hsv) for hsv in hsv_colors]
# # Данные
# categories = ["0.5","0.3","0.1"]
# values = abs(np.array([1.1426, 1.824, 12.145])-np.array([1.0743, 1.755, 12.128]))/np.array([1.0743, 1.755, 12.128])
# print(values*100)
# #values = values/values.max()
# # Сортируем данные по убыванию значений
# sorted_data = sorted(zip(values, categories), reverse=True)
# sorted_values = [x[0] for x in sorted_data]
# sorted_categories = [x[1] for x in sorted_data]
#
# # Создаем диаграмму
# plt.figure(figsize=(8, 6))
# plt.bar(sorted_categories, sorted_values, color=rgb_colors)
#
# # Добавляем подписи
# plt.ylabel('Податливость')
#
# # Показываем диаграмму
# plt.show()

"""123123123"""
# folder_path='C:/Users/mille/Documents/отчёт/difference_volf'
# filename_pattern = re.compile(r'volf=(\d+\.\d{3})\.txt')
# # Регулярное выражение для извлечения obnum из строки
# content_pattern = re.compile(r'Constraint\s*=\s*(-?\d+\.\d{3})')
#
# volf_dict = {}
# # Перебираем все файлы в папке
# for filename in os.listdir(folder_path):
#     match = filename_pattern.match(filename)
#     if match:
#         num = match.group(1)
#         file_path = os.path.join(folder_path, filename)
#         obnums = []
#
#         ob_count=0
#         # Читаем файл и извлекаем obnum
#         with open(file_path, 'r') as file:
#             for line in file:
#                 content_match = content_pattern.search(line)
#                 if content_match:
#                     obnum = np.log(float(content_match.group(1))+1)
#                     #if ob_count == 100: continue
#                     obnums.append(obnum)
#                     ob_count+=1
#
#         if obnums:
#             volf_dict[num] = np.array(obnums)
# x=np.arange(2,100,1)
#
# fig, ax = plt.subplots()
# for key in reversed(volf_dict):
#     print(key,'  ',len(volf_dict[key]))
#     ax.plot(x,volf_dict[key][2:],label=key)
# ax.legend()
# ax.set_ylim([0, 0.015])
# plt.show()
#

# folder_path='C:/Users/mille/Documents/отчёт/difference_volf'
# filename_pattern = re.compile(r'volf=(\d+\.\d{2})\.txt')
# # Регулярное выражение для извлечения obnum из строки
# content_pattern = re.compile(r'Constraint\s*=\s*(-?\d+\.\d{3})')
#
# volf_dict = {}
# name_dict = {}
# # Перебираем все файлы в папке
# for filename in os.listdir(folder_path):
#     print(filename)
#     match = filename_pattern.match(filename)
#
#     if match:
#         num = match.group(1)
#         if abs(0.25 - float(num)) < 0.05: num1 = str(0.25)
#         elif abs(0.1 - float(num)) < 0.05: num1 = str(0.1)
#         elif abs(0.4 - float(num)) < 0.05: num1 = str(0.4)
#         else: num1=num
#         name_dict[num]=num1
#         file_path = os.path.join(folder_path, filename)
#         obnums = []
#
#         # Читаем файл и извлекаем obnum
#         ob_count = 0
#         with open(file_path, 'r') as file:
#             for line in file:
#                 content_match = content_pattern.search(line)
#                 if content_match:
#                     #obnum = float(content_match.group(1))
#                     obnum = np.log(float(content_match.group(1)) + 1)
#                     #if ob_count == 100: continue
#                     obnums.append(obnum)
#                     ob_count+=1
#
#         if obnums:
#             volf_dict[num] = np.array(obnums)
#
#
# x=np.arange(1,199,1)
# fig, ax = plt.subplots()
# for key in volf_dict:
#     print(key,'  ',len(volf_dict[key]))
#     ax.plot(x,volf_dict[key][2:],label=name_dict[key])
# ax.legend(loc='lower right')
# ax.set_ylim([0, 0.2])
# plt.show()

