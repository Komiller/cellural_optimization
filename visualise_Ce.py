from model import model
import numpy as np
import matplotlib.pyplot as plt
from prepare import load_CHs, make_full_massive,load_dens
from mpl_toolkits.mplot3d import Axes3D
import json
import os
from sklearn.metrics import mean_squared_error


tmp=make_full_massive(load_CHs())


model=model('weights_box','scalers_box')

a={11:0,12:1,13:2,22:3,23:4,33:5,44:6,55:7,66:8}
"""
for numb in a:
    nn=a[numb]
    print(model.predict_value_CE([0.1],[0.1],[0.1])[:,nn])
    print(model.predict_value_CE([0.11],[0.1],[0.1])[:,nn])
    print(model.predict_value_CE([0.1],[0.11],[0.1])[:,nn])
    print(model.predict_value_CE([0.1],[0.1],[0.11])[:,nn])
    print('___')


"""

numbs=[11,12,13,22,23,33,44,55,66]
#numbs=[11]

print('started')
for numb in numbs:
    #if numb!=44:continue

    for iii in range(3):
        X = []
        Y = []
        for var in tmp:
            arr = list(var[0])
            if arr[0]==0.5 or arr[1]==0.5 or arr[2]==0.5:continue
            if round(arr[iii],4) == 0.1:
                X.append(arr)
                Y.append(var[1][numb//10-1,numb%10-1])
        x_train=np.array(X)
        y_train=np.array(Y)

        x = np.linspace(0, 0.5, 100)
        y = np.linspace(0, 0.5, 100)
        x, y = np.meshgrid(x, y)

        coords1 = [1, 0, 0]
        coords2 = [2, 2, 1]

        coords11 = [2,0,0]
        coords22 = [0, 2, 1]
        coords33 = [1, 1, 2]
        radii_for_model=[x.flatten(),y.flatten(),np.ones(len(x.flatten()))*0.1]

        z=model.predict_value_CE(radii_for_model[coords11[iii]],radii_for_model[coords22[iii]],radii_for_model[coords33[iii]])[:,a[numb]]
        z_diff_x=model.predict_sensitivies_CE(radii_for_model[coords11[iii]],radii_for_model[coords22[iii]],radii_for_model[coords33[iii]])[::3,a[numb]]
        z_diff_y = model.predict_sensitivies_CE(radii_for_model[coords11[iii]], radii_for_model[coords22[iii]],
                                                radii_for_model[coords33[iii]])[1::3, a[numb]]
        z_diff_z = model.predict_sensitivies_CE(radii_for_model[coords11[iii]], radii_for_model[coords22[iii]],
                                                radii_for_model[coords33[iii]])[2::3, a[numb]]





        print(f'nn = {numb}, fixed_axis = {iii}')

        # Постройте 3D-график
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        norm=z.max()
        ax.scatter(x_train[:, coords1[iii]], x_train[:, coords2[iii]], y_train/norm, color='black')
        ax.plot_surface(x, y, z.reshape(x.shape)/norm, cmap='viridis',
                         label=f'nn = {numb} axis = {coords1[iii]} {coords2[iii]}')


        """ax_diff = fig.add_subplot(2, 2, 2, projection='3d')
        ax_diff.plot_surface(x, y, z_diff_x.reshape(x.shape), cmap='viridis')
        ax_diff_y=fig.add_subplot(2, 2, 3, projection='3d')
        ax_diff_y.plot_surface(x, y, z_diff_y.reshape(x.shape), cmap='viridis')

        ax_diff_z = fig.add_subplot(2, 2, 4, projection='3d')
        ax_diff_z.plot_surface(x, y, z_diff_z.reshape(x.shape), cmap='viridis')"""



        # Настройте подписи осей
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel(f'C{numb}')
        #ax.set_title(f'C{numb}, r{iii+1} = const')

        """
        ax_diff.set_xlabel('X')
        ax_diff.set_ylabel('Y')
        ax_diff.set_zlabel('Z')
        ax_diff.set_title(f'd/dr1')


        ax_diff_y.set_xlabel('X')
        ax_diff_y.set_ylabel('Y')
        ax_diff_y.set_zlabel('Z')
        ax_diff_y.set_title(f'd/dr2')

        ax_diff_z.set_xlabel('X')
        ax_diff_z.set_ylabel('Y')
        ax_diff_z.set_zlabel('Z')
        ax_diff_z.set_title(f'd/dr3')
        """


        # Покажите график
        plt.show()

"""
"""
"""
x = np.linspace(0.1, 0.5, 100)
y = np.linspace(0.1, 0.5, 100)
x, y = np.meshgrid(x, y)
z=model.predict_value_density(x.flatten(),np.ones(len(x.flatten()))*0.1,y.flatten())
# Постройте 3D-график
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z.reshape(x.shape), cmap='viridis')
#ax.scatter(x_train[:,0], x_train[:,2], y_train, color='red')
# Настройте подписи осей
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Покажите график
plt.show()"""

"""elasticity_model=model('weights_zr','scalers_zr')
elasticity_model2=model('weights')
r=np.arange(-0.1,0.1,0.2/200)


Ce=elasticity_model.predict_value_CE(r,r,r)
Ce_fr=elasticity_model.predict_value_CE(r[0:-1],r[1::],r[1::])


Ced=elasticity_model.predict_sensitivies_CE(r,r,r)


cnum=[11,12,13,22,23,33,44,55,66]
for i in range(9):
    Ced_r = (Ce[1::,i] - Ce_fr[:,i]) / (0.2 / 200)
    plt.plot(r,Ce[:,i],color='red',label='Ce')
    plt.plot(r,Ced[::3,i],label=f'd/dr1')
    plt.plot(r,Ced[1::3,i],label=f'd/dr2')
    plt.plot(r,Ced[2::3,i],label=f'd/dr3')

    plt.plot(r[1::], Ced_r, label=f'd/dr1 разностная')
    plt.title(f'C{cnum[i]}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()"""