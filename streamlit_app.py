import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

st.set_page_config(layout="wide")
st.title("📈 LSTM Stock Price Forecast App")

uploaded_file = st.file_uploader("Upload your stock CSV file", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    try:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df[['Date', 'Open', 'High', 'Low', 'Volume', 'Close']].dropna()
        
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df[['Open', 'High', 'Low', 'Volume', 'Close']])
        
        def create_sequences(data, seq_len):
            X, y = [], []
            for i in range(len(data) - seq_len):
                X.append(data[i:i+seq_len, :])
                y.append(data[i+seq_len, -1])
            return np.array(X), np.array(y)
        
        sequence_length = 60
        X, y = create_sequences(scaled_data, sequence_length)
        
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        with st.spinner("Training the LSTM model..."):
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
                Dropout(0.2),
                LSTM(50),
                Dropout(0.2),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mean_squared_error')
            model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=0)

        # --- Prediction ---
        y_pred_scaled = model.predict(X_test)
        dummy_pred = np.zeros((len(y_pred_scaled), scaled_data.shape[1]))
        dummy_pred[:, -1] = y_pred_scaled.flatten()
        y_pred = scaler.inverse_transform(dummy_pred)[:, -1]

        dummy_test = np.zeros((len(y_test), scaled_data.shape[1]))
        dummy_test[:, -1] = y_test.flatten()
        y_test_actual = scaler.inverse_transform(dummy_test)[:, -1]

        # --- Evaluation Metrics ---
        mae = mean_absolute_error(y_test_actual, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))
        r2 = r2_score(y_test_actual, y_pred)
        mape = np.mean(np.abs((y_test_actual - y_pred) / y_test_actual)) * 100

        st.subheader("📊 Evaluation Metrics")
        st.markdown(f"- **MAE**: {mae:.2f}")
        st.markdown(f"- **RMSE**: {rmse:.2f}")
        st.markdown(f"- **R² Score**: {r2:.4f}")
        st.markdown(f"- **MAPE**: {mape:.2f}%")

        # --- Plot Actual vs Predicted ---
        st.subheader("📉 Actual vs Predicted Close Price")
        dates = df['Date'].iloc[train_size + sequence_length:train_size + sequence_length + len(y_test_actual)]
        fig1, ax1 = plt.subplots(figsize=(12, 5))
        ax1.plot(dates, y_test_actual, label='Actual', color='blue')
        ax1.plot(dates, y_pred, label='Predicted', color='orange')
        ax1.set_title("Actual vs Predicted Close Price")
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Price")
        ax1.legend()
        ax1.grid()
        st.pyplot(fig1)

        # --- Forecast Next 30 Days ---
        st.subheader("🔮 Next 30 Days Forecast")
        last_sequence = scaled_data[-sequence_length:]
        future_predictions = []

        current_seq = last_sequence.copy()
        for _ in range(30):
            next_scaled = model.predict(current_seq.reshape(1, sequence_length, 5), verbose=0)[0][0]
            dummy = np.zeros((1, scaled_data.shape[1]))
            dummy[0, -1] = next_scaled
            next_actual = scaler.inverse_transform(dummy)[0, -1]
            future_predictions.append(next_actual)

            next_input = current_seq[-1].copy()
            next_input[-1] = next_scaled
            current_seq = np.vstack((current_seq[1:], next_input))

        last_date = df['Date'].iloc[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30, freq='B')
        forecast_df = pd.DataFrame({'Date': future_dates, 'Predicted Close': future_predictions})

        fig2, ax2 = plt.subplots(figsize=(10, 5))
        ax2.plot(future_dates, future_predictions, marker='o', color='green')
        ax2.set_title("Next 30 Days Forecast")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Predicted Close Price")
        ax2.grid()
        st.pyplot(fig2)

        st.dataframe(forecast_df)

    except Exception as e:
        st.error(f"Error processing file: {e}")
