

import numpy as np
import pandas as pd
import tensorflow as tf
import yfinance as yf

from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, LSTM

ticker_symbol = "TATASTEEL.NS"
ticker = yf.Ticker(ticker_symbol)
history = ticker.history(period="3mo",interval='1d')


ticker.history_metadata

history.Close.plot()
plt.show()

ticker.cash_flow.index

def split_sequence(sequence, n_steps):
  X, Y = [], []
  n = len(sequence)
  for i in range(n):
    end_idx = i + n_steps
    if end_idx > n - 1:
      break;
    seq_x, seq_y = sequence.iloc[i:end_idx], sequence.iloc[end_idx]
    X.append(seq_x)
    Y.append(seq_y)
  return np.array(X), np.array(Y)

n_steps = 5
X, y = split_sequence(history['Close'],n_steps)

"""### Vanilla LSTM (Plain LSTM)"""

n_features = 1
model = Sequential()
model.add(Input(shape=(n_steps,n_features)))
model.add(LSTM(100, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
X = X.reshape((X.shape[0],X.shape[1],n_features))

s = int(len(X)*.8) # 80%-20% split as per Pareto rule
x_train, x_test = X[:s], X[s:]
y_train, y_test = y[:s], y[s:]

model.fit(x_train,y_train,epochs=200,verbose=0)

y_pred = []
for i in x_test:
    x_input = np.array(i)
    x_input = x_input.reshape((1,n_steps,n_features))
    y_pred.append(model.predict(x_input,verbose=0))

y_pred = np.array(y_pred)

plt.plot(y)
# plt.plot(np.append(y_train,y_pred))
plt.plot(range(s,s+len(y_pred)),y_pred.ravel())
plt.legend(["original","predicted"])
plt.show()

y_pred = []
x_pred = x_train[-1].ravel()

for i in range(len(x_test)):
    x_input = x_pred[-n_steps:]
    x_input = x_input.reshape((1,n_steps,n_features))
    y_pred.append(model.predict(x_input,verbose=0))
    x_pred = np.append(x_pred, y_pred[-1])

y_pred = np.array(y_pred)

plt.plot(y)
# plt.plot(np.append(y_train,y_pred))
plt.plot(range(s,s+len(y_pred)),y_pred.ravel())
plt.legend(["original","predicted"])
plt.show()

"""### Stacked LSTM"""

model = Sequential()
model.add(Input(shape=(n_steps,n_features)))
model.add(LSTM(100, activation='relu', return_sequences=True))
model.add(LSTM(150, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(x_train,y_train,epochs=200,verbose=0)

y_pred = []
for i in x_test:
    x_input = np.array(i)
    x_input = x_input.reshape((1,n_steps,n_features))
    y_pred.append(model.predict(x_input,verbose=0))

y_pred = np.array(y_pred)

plt.plot(y)
# plt.plot(np.append(y_train,y_pred))
plt.plot(range(s,s+len(y_pred)),y_pred.ravel())
plt.legend(["original","predicted"])
plt.show()

y_pred = []
x_pred = x_train[-1].ravel()

for i in range(len(x_test)):
    x_input = x_pred[-n_steps:]
    x_input = x_input.reshape((1,n_steps,n_features))
    y_pred.append(model.predict(x_input,verbose=0))
    x_pred = np.append(x_pred, y_pred[-1])
    # print(y_pred[-1])

y_pred = np.array(y_pred)

plt.plot(y)
# plt.plot(np.append(y_train,y_pred))
plt.plot(range(s,s+len(y_pred)),y_pred.ravel())
plt.legend(["original","predicted"])
plt.show()

"""### Bidirectional LSTM"""

from tensorflow.keras.layers import Bidirectional

model = Sequential()
model.add(Input(shape=(n_steps,n_features)))
model.add(Bidirectional(LSTM(150, activation='relu')))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(x_train,y_train,epochs=200,verbose=1)

y_pred = []
for i in x_test:
    x_input = np.array(i)
    x_input = x_input.reshape((1,n_steps,n_features))
    y_pred.append(model.predict(x_input,verbose=0))

y_pred = np.array(y_pred)

plt.plot(y)
# plt.plot(np.append(y_train,y_pred))
plt.plot(range(s,s+len(y_pred)),y_pred.ravel())
plt.legend(["original","predicted"])
plt.show()

y_pred = []
x_pred = x_train[-1].ravel()

for i in range(len(x_test)):
    x_input = x_pred[-n_steps:]
    x_input = x_input.reshape((1,n_steps,n_features))
    y_pred.append(model.predict(x_input,verbose=0))
    x_pred = np.append(x_pred, y_pred[-1])

y_pred = np.array(y_pred)

plt.plot(y)
# plt.plot(np.append(y_train,y_pred))
plt.plot(range(s,s+len(y_pred)),y_pred.ravel())
plt.legend(["original","predicted"])
plt.show()