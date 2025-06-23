
import numpy as np
import json
from prepare import load_CHs, make_full_massive,load_dens
from sklearn.preprocessing import MinMaxScaler
import pickle
import matplotlib.pyplot as plt

tmp=make_full_massive(load_CHs())

for i in range(1,7):
    for j in range(i,7):
        numb=int(str(i)+str(j))
        for ikl in range(9):
            X = []
            Y = []
            for var in tmp:
                arr = list(var[0])
                if int(round(arr[0], 3) * 1000) % 50 != 0: continue
                if int(round(arr[1], 3) * 1000) % 50 != 0: continue
                if int(round(arr[2], 3) * 1000) % 50 != 0: continue
                X.append(arr)
                Y.append(var[1][numb//10-1,numb%10-1])

        x_train=np.array(X)
        y_train=np.array(Y)
        scaler = MinMaxScaler(feature_range=(-1, 1))
        y_train = scaler.fit_transform(y_train.reshape(-1, 1)).flatten()


        with open(f'scalers_PAPC/scaler{numb}.pkl', 'wb') as f:
            pickle.dump(scaler, f)



        # with open(f'PAPC_js/Ch{numb}.json', 'w') as f:
        #     json.dump({'inputs':x_train.tolist(),'outputs': y_train.tolist()}, f)

# from itertools import permutations
# tmp=load_dens()
# X = []
# Y = []
# for var in tmp:
#     arr = list(tmp[var][0])
#     arr_perm = np.array(list(permutations(arr)))
#     for arr in arr_perm:
#         X.append(arr)
#         Y.append(tmp[var][1])
# x_train=np.array(X)
# y_train=np.array(Y)
# print(x_train.tolist())
# print(y_train)
# with open(f'PAPC_js/densities.json', 'w') as f:
#     json.dump({'inputs': x_train.tolist(), 'outputs': y_train.tolist()}, f)

