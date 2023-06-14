import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# set seed
np.random.seed(0)
tf.random.set_seed(0)

# load training dataset
train_df = pd.read_csv("training.csv")
train_x = np.array(train_df.loc[:,"V2":"sig_eff"])
train_y = np.array(train_df.loc[:,"loading"])

# load validation dataset
valid_df = pd.read_csv("validation.csv")
valid_x = np.array(valid_df.loc[:,"V2":"sig_eff"])
valid_y = np.array(valid_df.loc[:,"loading"])

# scale inputs/predictors
scaler = MinMaxScaler(feature_range=(0,1))
train_x = scaler.fit_transform(train_x)
valid_x = scaler.transform(valid_x)

# define a nueral network model
model = Sequential()
model.add(Dense(50, input_dim=31, kernel_initializer='glorot_normal', activation='sigmoid'))
model.add(Dense(50, kernel_initializer='glorot_normal', activation='sigmoid'))
model.add(Dense(20, kernel_initializer='glorot_normal', activation='sigmoid'))
model.add(Dense(20, kernel_initializer='glorot_normal', activation='sigmoid'))
model.add(Dense(1, kernel_initializer='glorot_normal', activation='linear'))

# compile model
model.compile(loss='mae', optimizer=Adam(learning_rate=0.0001), metrics=['mse','mape'])

# define an early stopping condition
es = EarlyStopping(monitor='val_loss', patience=20, verbose=1)

# fit model
history = model.fit(train_x, train_y, validation_data=(valid_x,valid_y), epochs=500, batch_size=50, verbose=2, callbacks=[es])

# save model
model.save('gas_loading_prediction-practice.h5')

# save learning curve
fig = plt.figure(dpi=150)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training','validation'])
plt.ylabel('Loss')
plt.xlabel('Epoch')
fig.savefig('learning_curve.png')
