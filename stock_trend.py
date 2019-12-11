import numpy as np
import pandas as pd
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

#Loading and Preprocessing Data
prices_dataset_train = pd.read_csv('train.csv') #DATA  ACQUIRED FROM YAHOO FINANCE
prices_dataset_test = pd.read_csv('test.csv')

#select only the required column
trainingset =prices_dataset_train.iloc[:,5:6].values  #using adjusted prices
testset = prices_dataset_test.iloc[:,5:6].values

#trainingset.fillna(trainingset.mean())
#trainingset.mean()

################## min-max normalization
min_max_scaler = MinMaxScaler(feature_range =(0,1))
scaled_trainingset = min_max_scaler.fit_transform(trainingset)

#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#scaled_trainingset = sc.fit_transform(trainingset)


x_train = []
y_train = []

for i in range(40 , 1475):
    x_train.append(scaled_trainingset[i-40:i, 0])
    y_train.append(scaled_trainingset[i , 0])
    
x_train = np.array(x_train)
y_train = np.array(y_train)

x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))

#building the model

model = Sequential()
model.add(LSTM (units = 100,return_sequences = True, input_shape = (x_train.shape[1],1)))
model.add(Dropout(0.5))
model.add(LSTM (units = 50,return_sequences = True))
model.add(Dropout(0.3))
model.add(LSTM (units =50))
model.add(Dropout(0.3))
model.add(Dense(units = 1))


model.compile(optimizer = 'adam',loss = 'mse')
model.fit(x_train, y_train,epochs = 100, batch_size =32)


dataset_total = pd.concat((prices_dataset_train['Adj Close'],prices_dataset_test['Adj Close']),axis = 0)
inputs = dataset_total[len(dataset_total)-len(prices_dataset_test)-40:].values
inputs = inputs.reshape(-1,1)


inputs = min_max_scaler.transform(inputs)
x_test = []

for i in range(40,len(prices_dataset_test)+40):
    x_test.append(inputs[i-40:i , 0])
    
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))

pred = model.predict(x_test)
pred = min_max_scaler.inverse_transform(pred)

plt.plot(testset, color = 'blue', label = 'actual_prices')
plt.plot(pred, color = 'green',label = 'LSTM_predictions')
plt.title('RNN_trend_prediction')
plt.xlabel('time')
plt.ylabel('price')
plt.legend()
plt.show()


















