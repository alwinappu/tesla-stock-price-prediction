import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Tesla Stock Price Prediction",
    layout="wide"
)

st.title("Tesla Stock Price Prediction")
st.markdown("Time-Series Forecasting using Deep Learning")
st.markdown("---")

# Sidebar
model_choice = st.sidebar.selectbox(
    "Select Model",
    ["SimpleRNN", "LSTM"]
)

days = st.sidebar.radio(
    "Prediction Horizon (Days)",
    [1, 5, 10]
)

@st.cache_data
def load_data():
    df = pd.read_csv("TSLA.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    return df

df = load_data()
close_data = df[["Close"]]

scaler = MinMaxScaler()
scaled_close = scaler.fit_transform(close_data)

WINDOW_SIZE = 60
last_sequence = scaled_close[-WINDOW_SIZE:]

@st.cache_resource
def load_model(model_name):
    if model_name == "SimpleRNN":
        return tf.keras.models.load_model("simple_rnn_model.h5")
    else:
        return tf.keras.models.load_model("lstm_model.h5")

model = load_model(model_choice)

def predict_future(model, seq, days):
    preds = []
    current_seq = seq.copy()

    for _ in range(days):
        pred = model.predict(
            current_seq.reshape(1, WINDOW_SIZE, 1),
            verbose=0
        )[0][0]
        preds.append(pred)
        current_seq = np.append(current_seq[1:], pred)

    return scaler.inverse_transform(np.array(preds).reshape(-1, 1))

if st.button("Predict"):
    predictions = predict_future(model, last_sequence, days)

    cols = st.columns(len(predictions))
    for i, col in enumerate(cols):
        col.metric(f"Day {i+1}", f"${predictions[i][0]:.2f}")

    fig, ax = plt.subplots()
    ax.plot(predictions, marker="o")
    ax.set_title(f"{model_choice} – {days} Day Forecast")
    ax.set_xlabel("Days Ahead")
    ax.set_ylabel("Price")
    ax.grid(True)

    st.pyplot(fig)
