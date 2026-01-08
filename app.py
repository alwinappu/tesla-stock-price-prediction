import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Tesla Stock Price Prediction",
    page_icon="📈",
    layout="wide"
)

# ---------------- TITLE ----------------
st.title("Tesla Stock Price Prediction")
st.markdown("Deep Learning based Time-Series Forecasting using SimpleRNN and LSTM")
st.markdown("---")

# ---------------- SIDEBAR ----------------
st.sidebar.header("Configuration")

model_choice = st.sidebar.selectbox(
    "Select Model",
    ["SimpleRNN", "LSTM"]
)

days = st.sidebar.radio(
    "Prediction Horizon (Days)",
    [1, 5, 10]
)

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    df = pd.read_csv("TSLA.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    return df

df = load_data()
close_data = df[["Close"]]

# ---------------- SCALE DATA ----------------
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_close = scaler.fit_transform(close_data)

WINDOW_SIZE = 60
last_sequence = scaled_close[-WINDOW_SIZE:]

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_dl_model(model_name):
    if model_name == "SimpleRNN":
        return tf.keras.models.load_model("simple_rnn_model.h5")
    else:
        return tf.keras.models.load_model("lstm_model.h5")

model = load_dl_model(model_choice)

# ---------------- PREDICTION FUNCTION ----------------
def predict_future(model, seq, days, window):
    preds = []
    current_seq = seq.copy()

    for _ in range(days):
        pred = model.predict(
            current_seq.reshape(1, window, 1),
            verbose=0
        )[0][0]

        preds.append(pred)
        current_seq = np.append(current_seq[1:], pred)

    return scaler.inverse_transform(
        np.array(preds).reshape(-1, 1)
    )

# ---------------- MAIN UI ----------------
st.subheader("Future Closing Price Prediction")

if st.button("Predict"):
    predictions = predict_future(model, last_sequence, days, WINDOW_SIZE)

    # Display predictions
    cols = st.columns(len(predictions))
    for i, col in enumerate(cols):
        col.metric(
            label=f"Day {i+1}",
            value=f"${predictions[i][0]:.2f}"
        )

    st.markdown("---")

    # Plot predictions
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(predictions, marker="o", linewidth=2)
    ax.set_title(f"{model_choice} – {days}-Day Forecast")
    ax.set_xlabel("Days Ahead")
    ax.set_ylabel("Predicted Closing Price ($)")
    ax.grid(True)

    st.pyplot(fig)
