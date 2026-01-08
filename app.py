import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Tesla Stock Price Prediction",
    page_icon="📈",
    layout="wide"
)

# ---------------- TITLE ----------------
st.markdown(
    """
    <h1 style='text-align: center;'>🚗 Tesla Stock Price Prediction</h1>
    <p style='text-align: center; font-size:18px;'>
    Deep Learning using SimpleRNN & LSTM
    </p>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# ---------------- SIDEBAR ----------------
st.sidebar.header("Model Configuration")

st.sidebar.markdown(
    """
    This application predicts **Tesla's closing stock price**
    using deep learning time-series models.
    """
)

model_choice = st.sidebar.selectbox(
    "Choose Model",
    ["SimpleRNN", "LSTM"]
)

days = st.sidebar.radio(
    "Prediction Horizon",
    [1, 5, 10]
)

st.sidebar.markdown("---")
st.sidebar.info(
    """
    **Dataset:** Tesla Historical Prices  
    **Target:** Closing Price  
    **Window Size:** 60 days
    """
)

# ---------------- LOAD DATA ----------------
df = pd.read_csv("TSLA.csv")
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

close_data = df[['Close']]

# ---------------- SCALE DATA ----------------
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_close = scaler.fit_transform(close_data)

WINDOW_SIZE = 60
last_sequence = scaled_close[-WINDOW_SIZE:]

# ---------------- LOAD MODEL ----------------
if model_choice == "SimpleRNN":
    model = load_model("simple_rnn_model.h5")
else:
    model = load_model("lstm_model.h5")

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
st.subheader("🔮 Predict Future Closing Prices")

if st.button("Run Prediction"):
    predictions = predict_future(model, last_sequence, days, WINDOW_SIZE)

    # KPI Cards
    cols = st.columns(len(predictions))
    for i, col in enumerate(cols):
        col.metric(
            label=f"Day {i+1}",
            value=f"${predictions[i][0]:.2f}"
        )

    st.markdown("---")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(predictions, marker='o', linewidth=2)
    ax.set_title(f"{model_choice} – {days}-Day Forecast")
    ax.set_xlabel("Days Ahead")
    ax.set_ylabel("Predicted Price ($)")
    ax.grid(True)

    st.pyplot(fig)

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown(
    """
    <p style='text-align: center; color: grey;'>
    Developed by Appu | MSc Data Science | Deep Learning Project
    </p>
    """,
    unsafe_allow_html=True
)
