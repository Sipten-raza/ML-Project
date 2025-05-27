# app.py

import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(layout="centered")
st.title('📈 Stock Market Predictor')

# Input from user
stock = st.text_input('Enter Stock Symbol', 'GOOG')
start = '2012-01-01'
end = '2022-12-31'

# Load data
data = yf.download(stock, start, end)

if data.empty:
    st.error("Invalid stock symbol or no data found. Please try a different symbol.")
    st.stop()

# Display raw stock data
st.subheader('Stock Data')
st.write(data.tail())

# Train-test split
data_train = pd.DataFrame(data.Close[0:int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80):])
scaler = MinMaxScaler(feature_range=(0,1))

past_100_days = data_train.tail(100)
data_test = pd.concat([past_100_days, data_test], ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)

# Moving averages plots
st.subheader('Price vs MA50')
ma_50 = data.Close.rolling(50).mean()
fig1 = plt.figure(figsize=(10,4))
plt.plot(ma_50, 'r', label='MA50')
plt.plot(data.Close, 'g', label='Closing Price')
plt.legend()
st.pyplot(fig1)

st.subheader('Price vs MA50 vs MA100')
ma_100 = data.Close.rolling(100).mean()
fig2 = plt.figure(figsize=(10,4))
plt.plot(ma_50, 'r', label='MA50')
plt.plot(ma_100, 'b', label='MA100')
plt.plot(data.Close, 'g', label='Closing Price')
plt.legend()
st.pyplot(fig2)

st.subheader('Price vs MA100 vs MA200')
ma_200 = data.Close.rolling(200).mean()
fig3 = plt.figure(figsize=(10,4))
plt.plot(ma_100, 'r', label='MA100')
plt.plot(ma_200, 'b', label='MA200')
plt.plot(data.Close, 'g', label='Closing Price')
plt.legend()
st.pyplot(fig3)

# Prepare test data
x = []
y = []
for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i, 0])

x, y = np.array(x), np.array(y)

# Load model and make predictions
model = load_model("Stock Predictions Model.keras")
predictions = model.predict(x)

# Rescale back
scale = scaler.scale_[0]
predictions = predictions / scale
y = y / scale

# Prediction plot
st.subheader('📉 Original vs Predicted Price')
fig4 = plt.figure(figsize=(10,4))
plt.plot(y, 'r', label='Original Price')
plt.plot(predictions, 'g', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
st.pyplot(fig4)
