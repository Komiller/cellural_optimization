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


elem_id=296
np.set_printoptions(precision=3, threshold=20)
stress_amplification=np.zeros((6,6))


diffs1=[]
diffs2=[]
iar=[]

for i in range(1,11):
    if i==9:continue
    op2_filename = f'D:/tests/hom/{i}-0000.op2'
    op2 = read_op2(op2_filename, build_dataframe=True, debug=False)
    en = op2.op2_results.strain_energy.chexa_strain_energy[1]
    en1 = en.data[0, -1, 0]
    en = op2.op2_results.strain_energy.chexa_strain_energy[2]
    en11 = en.data[0, -1, 0]


    op2_filename = f'D:/tests/det/{i}-0000.op2'
    op2 = read_op2(op2_filename, build_dataframe=True, debug=False)
    en = op2.op2_results.strain_energy.chexa_strain_energy[1]
    en2 = en.data[0, -1, 0]
    en = op2.op2_results.strain_energy.chexa_strain_energy[2]
    en21 = en.data[0, -1, 0]

    diffs1.append((abs(en1 - en2) / en2)*100)


    print(en11,en21)
    diffs2.append((abs(en11 - en21) / en21)*100)
    iar.append(i)
fig, ax = plt.subplots()
ax.plot(iar,diffs1,color='red',label='Изгиб')
ax.plot(iar,diffs2,color='black',label='Сжатие')
print(diffs2)

from matplotlib.ticker import FuncFormatter
def to_percent(x, pos):
    return f"{x:.0f}%"
formatter = FuncFormatter(to_percent)
ax.yaxis.set_major_formatter(formatter)
plt.grid(True)
plt.xlabel("N")  # Подпись оси X
plt.ylabel("$\Delta$ E")  # Подпись оси Y
plt.legend()

plt.show()






# stress = op2.op2_results.stress.chexa_stress[1]
#
# for i in range(6):
#     for j in range(6):
#
#         stress = op2.op2_results.stress.chexa_stress[i+1]
#         element_node = stress.element_node
#         elements = element_node[:, 0]
#         elm_id = np.where(element_node[:, 0] == elem_id)[0][0]
#         stress_amplification[j,i]=stress.data[0,elm_id,j]
#
# print(stress_amplification.dot(np.array([0.003887,0.07093,-0.001389,-0.0001788,-0.00006873,0.3062])))
#print(stress_amplification.dot(np.array([0,0,0,0,0,1])))