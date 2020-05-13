import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers
import pandas as pd
import matplotlib.pyplot as plt

df1 = pd.read_csv("./testcsv.csv", encoding='utf-8')

sunshine = np.array(df1.sunshine)
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
        #in_data = data1[i:(i+window_size)] + data2[i:(i+window_size)] + data3[i:(i+window_size)]
        in_data = np.concatenate((data1[i:(i+window_size)], data2[i:(i+window_size)], data3[i:(i+window_size)]), axis=0)
        dataX.append(in_data)
        dataY.append(out[i + window_size])
    return np.array(dataX), np.array(dataY)

window_size = 5

input = pgen
output = pgen
#x_train, y_train = create_dataset(input, window_size)
x_train, y_train = create_dataset_multi(suntime, sunshine, pgen_norm, pgen_norm, window_size)
#__, y_train = create_dataset(output, window_size)

Model = Sequential()
Model.add(Dense(128, input_dim=window_size * 3, activation='relu'))
Model.add(Dense(64, activation='relu'))
Model.add(Dense(64, activation='relu'))
Model.add(Dense(1, activation='relu'))
Model.compile(loss='mean_squared_error', optimizer=optimizers.Adam(lr=0.0001))
Model.fit(x_train, y_train, epochs=100, batch_size=64, verbose=1)

pred = Model.predict(x_train)


plt.plot(pred, 'b')
plt.plot(y_train, 'y', linestyle=':')
plt.show()

# Need another data for validation