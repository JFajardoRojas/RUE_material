import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# load model
model = load_model('gas_loading_prediction-practice.h5')

# load training dataset
train_df = pd.read_csv("training.csv")
train_x = np.array(train_df.loc[:,"V2":"sig_eff"])
train_y = np.array(train_df.loc[:,"loading"])

# load test dataset
test_df = pd.read_csv("test.csv")
test_x = np.array(test_df.loc[:,"V2":"sig_eff"])
test_y = np.array(test_df.loc[:,"loading"])

# scale inputs/predictors
scaler = MinMaxScaler(feature_range=(0,1))
train_x = scaler.fit_transform(train_x)
test_x = scaler.transform(test_x)

# gas loading prediction
pred_y = model.predict(test_x, verbose=0)

# plot
fig = plt.figure(dpi=150)
plt.plot(test_y, pred_y, 'b.')
plt.plot([-5,120],[-5,120], 'k-')
plt.xlim(-5,120)
plt.ylim(-5,120)
plt.xlabel("GCMC Loading")
plt.ylabel("ML Prediction")
fig.savefig('model-test.png')

# plot by loadings of different gases in CoRE MOFs
core_df = pd.read_csv("test.csv") # You can also use here the CoREMOFs test file to 

def plot(gas):

    load_df = core_df[core_df['ads'] == gas]
    load_x = np.array(load_df.loc[:,"V2":"sig_eff"])
    load_x = scaler.transform(load_x)
    load_y = np.array(load_df.loc[:,"loading"])

    pred_y = model.predict(load_x, verbose=0)
    fig = plt.figure(dpi=150)
    plt.plot(load_y, pred_y, 'b.')
    plt.plot([-5,120],[-5,120], 'k-')
    plt.xlim(-5,80)
    plt.ylim(-5,80)
    plt.xlabel("GCMC Loading")
    plt.ylabel("ML Prediction")
    fig.savefig(gas + '.png')

for gas in ['argon','methane','krypton','xenon','N2','ethane']:
    plot(gas)
