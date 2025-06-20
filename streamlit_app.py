import numpy as np
import pandas as pd
import pandas_datareader as data
import warnings
import streamlit as st
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (9, 5)
matplotlib.rcParams['figure.facecolor'] = '#00000000'
from keras.models import load_model
warnings.simplefilter(action='ignore', category=FutureWarning)
from datetime import datetime   

start = pd.to_datetime(['2010-01-01']).astype(int)[0]//10**9 # convert to unix timestamp.
end = pd.to_datetime(['2021-12-31']).astype(int)[0]//10**9 # convert to unix timestamp.

st.title('Stock / Crypto Analysis')
st.markdown("![Alt Text](https://c.tenor.com/jw92b2HUuTAAAAAC/stonks-stocks.gif)")

stock_ticker = st.text_input('Enter The Ticker From Yahoo-Finance eg. AAPL, RELIANCE.NS, BTC-USD','AAPL')
url = 'https://query1.finance.yahoo.com/v7/finance/download/' + stock_ticker + '?period1=' + str(start) + '&period2=' + str(end) + '&interval=1d&events=history'
df = pd.read_csv(url)

st.subheader(stock_ticker + ' Data From 2010 To 2021')
st.write(df)

# Describing Data
st.subheader(stock_ticker + ' Statistical Data')
st.write(df.describe())

#Visualisation

st.subheader('Closing Price v/s Time Chart With 100 SMA And 200 SMA')
st.write("A 100-day and 200-day Moving Average (MA) is the average of closing prices of the previous 100 days and 200 days respectively")
st.write("As Per Market Experts -> Buy signal appear when SMA-100 line cut SMA-200 line in its way upward")

ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(df.Close, 'b', label = 'Closing Price')
plt.plot(ma100, 'r', label = '100 SMA')
plt.plot(ma200, 'g', label = '200 SMA')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig)

# Splitting Data Into Training And Testing
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0,1))
data_training_array = scaler.fit_transform(data_training)

#Loading Model
model = load_model('keras_model.h5')

#Testing Model
past_100_days = data_training.tail(100)
final_df = past_100_days.append(data_testing, ignore_index = True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
  x_test.append(input_data[i - 100: i])
  y_test.append(input_data[i,0])

x_test, y_test = np.array(x_test), np.array(y_test)
y_pred = model.predict(x_test)
y_predicted = y_pred /scaler.scale_
y_test = y_test /scaler.scale_
st.subheader('Original Price v/s Predicted Price')
fig2 = plt.figure(figsize = (12,6))
plt.plot(y_test, 'b', label = 'Original Price');
plt.plot(y_predicted, 'r', label = 'Predicted Price');
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)

st.write("This Is Just For Educational Purpose And No Way A Financial Advice.")
st.write("Made With ❤️ By [Anish](https://github.com/anishaga)")
