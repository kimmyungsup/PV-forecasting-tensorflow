import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras import optimizers
import pandas as pd
import matplotlib.pyplot as plt

df1 = pd.read_csv("./testcsv.csv", encoding='utf-8')

sunshine = np.array(df1.sunshine_norm)
temperature = np.array(df1.tem_norm)
suntime = np.array(df1.sun_norm)
pgen = np.array(df1.Pgen)
pgen_norm = np.array(df1.Pgen_norm)

'''
def create_dataset(seqdata, window_size=1):
    dataX, dataY = [], []
    for i in range(len(seqdata)-window_size):
        dataX.append(seqdata[i:(i+window_size)])
        dataY.append(seqdata[i + window_size])
    return np.array(dataX), np.array(dataY)
'''

def create_dataset_multi(data1, data2, data3, out, window_size=1):
    dataX, dataY = [], []
    for i in range(len(data1)-window_size):
        #in_data = np.concatenate((data1[i:(i+window_size)], data2[i:(i+window_size)], data3[i:(i+window_size)]), axis=0) #MLP
        in_data = np.array((data1[i:(i + window_size)], data2[i:(i + window_size)], data3[i:(i + window_size)])) #LSTM
        dataX.append(in_data)
        dataY.append(out[i + window_size])
    return np.array(dataX), np.array(dataY)

window_size = 5
n_input = 3 # feature size
x_train, y_train = create_dataset_multi(suntime, sunshine, pgen_norm, pgen_norm, window_size)

#x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1) # single data
x_train = x_train.reshape(x_train.shape[0], window_size, n_input)


Model = Sequential()
Model.add(LSTM(64, input_shape=(x_train.shape[1], x_train.shape[2]), return_sequences=False))  # multi recurrent latyer -> return_sequences=True
Model.add(Dense(32, activation='relu'))
Model.add(Dense(1, activation='relu'))

Model.compile(loss='mae', optimizer=optimizers.Adam(lr=0.0001))
Model.fit(x_train, y_train, epochs=100, batch_size=64, verbose=1)

pred = Model.predict(x_train)

plt.plot(pred, 'b')
plt.plot(y_train, 'y', linestyle=':')
plt.show()

# Need another data for validation