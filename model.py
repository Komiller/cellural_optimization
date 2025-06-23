import scipy.io
import numpy as np
from typing import Annotated
from numpy.typing import NDArray
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import pickle
isotropic=np.array([[269.2308,115.3846,115.3846,0,0,0],
                    [115.3846,269.2308,115.3846,0,0,0],
                    [115.3846,115.3846,269.2308,0,0,0],
                    [0,0,0,76.92308,0,0],
                    [0,0,0,0,76.92308,0],
                    [0,0,0,0,0,76.92308]])
lowest_c=np.ones(9)*1e-3
class model:
    numers=[11,12,13,22,23,33,44,55,66,99]
    def __init__(self,weights_path,scaler_path='scalers'):
        self.models=[]
        self.scalers=[]
        self.scalars_diffs=[]
        count=0
        for num in self.numers:
            if num!=99:
                data = scipy.io.loadmat(f'{weights_path}/fitnet_weights_{num}.mat')
                with open(f'{scaler_path}/scaler{num}.pkl', 'rb') as f:
                    self.scalers.append(pickle.load(f))
                    self.scalars_diffs.append(self.scalers[count].data_max_ - self.scalers[count].data_min_)
                    count+=1
            else:
                data = scipy.io.loadmat(f'{weights_path}/density.mat')

            IW = data['IW']
            LW = data['LW']
            b1 = data['b1']
            b2 = data['b2']

            model = Sequential()
            neuron_number=len(LW.T)
            model.add(Dense(neuron_number, input_dim=3, activation='tanh'))
            model.add(Dense(1, activation='linear'))
            model.layers[0].set_weights([np.array(IW.T), np.array(b1.flatten())])
            model.layers[1].set_weights([np.array(LW.T), np.array(b2.flatten())])
            self.models.append(model)



    def predict_value_CE(self,r1:[float],r2:[float],r3:[float]) -> Annotated[NDArray[np.float64], "shape=(len(r),9,)"]:
        """
        predicts Ce of PA cell
        :param r1: first strut radiai
        :param r2: second strut radiai
        :param r3: third strut radiai
        :return: array of Ce values for each cell
        """
        prediction=[]
        input=np.array([r1,r2,r3]).T
        count=0
        for model in self.models[:-1]:
            prediction.append(np.clip(self.scalers[count].inverse_transform(model.predict(input,verbose=0).reshape(-1,1)).flatten(),lowest_c[count],None))
            count+=1
        return np.array(prediction).T.reshape(len(r1),9)

    def predict_value_density(self,r1:[float],r2:[float],r3:[float]) -> Annotated[NDArray[np.float64], "shape=(len(r),)"]:
        input=np.array([r1,r2,r3]).T
        return np.clip(self.models[-1].predict(input,verbose=0),0,None)

    def predict_sensitivies_CE(self,r1:[float],r2:[float],r3:[float]) -> Annotated[NDArray[np.float64], "shape=(3*len(r),9,)"]:
        """
        predicts dCe/dri of PA cell
        :param r1: first strut radiai
        :param r2: second strut radiai
        :param r3: third strut radiai
        :return: array of dCe/dri
        """
        prediction = []
        input = np.array([r1, r2, r3]).T
        count=0
        for model in self.models[:-1]:
            input_value = tf.constant(input, dtype=tf.float32)
            with tf.GradientTape() as tape:
                tape.watch(input_value)
                output = model(input_value)
            gradient = tape.gradient(output, input_value)
            prediction.append(gradient*self.scalars_diffs[count])
        return np.concatenate(np.array(prediction).transpose(1,0,2),1).T

    def predict_sensitivies_density(self,r1:[float],r2:[float],r3:[float]) -> Annotated[NDArray[np.float64], "shape=(3*len(r),)"]:
        input = np.array([r1, r2, r3]).T
        model=self.models[-1]
        input_value = tf.constant(input, dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(input_value)
            output = model(input_value)
        gradient = tape.gradient(output, input_value)
        return np.array(gradient)

    def predict_value_CE_debug(self,r1:[float],r2:[float],r3:[float]):
        prediction=[]
        input=np.array([r1,r2,r3]).T
        ones=np.ones(len(r1))
        for i in range(0,6):
            for j in range(i,6):
                if isotropic[i][j] == 0: continue
                prediction.append(isotropic[i][j]*np.ones(len(r1)))
        return np.array(prediction).T.reshape(len(r1),9)

    def predict_sensitivies_debug(self,r1:[float],r2:[float],r3:[float]):
        prediction = []
        input = np.array([r1]).T
        ones = np.ones((len(r1),3))
        input_value = tf.constant(input, dtype=tf.float32)
        tmp1=np.array([1,0,0])
        tmp2 = np.array([0,1,0])
        tmp3 = np.array([0,0,1])
        for i in range(0,6):
            for j in range(i,6):
                if isotropic[i][j] == 0: continue
                if i==j and i<3:
                    with tf.GradientTape() as tape:
                        tape.watch(input_value)
                        output = self.model11(input_value)
                        gradient = tape.gradient(output, input_value)
                    if i==0: tmp=tmp1
                    if i==1: tmp=tmp2
                    if i==2: tmp=tmp3
                    prediction.append(np.kron(np.array(gradient),tmp))
                else: prediction.append((0*ones))
        return np.concatenate(np.array(prediction).transpose(1,0,2),1).T
