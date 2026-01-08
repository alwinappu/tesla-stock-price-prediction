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

st.title("🚗 Tesla Stock Price Prediction")
st.markdown("**Time-Series Forecasting using Deep Learning**")
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

@st.cache_resource
def build_model(model_type, window_size=60):
    """Build a simple model on the fly"""
    model = tf.keras.Sequential()
    
    if model_type == "SimpleRNN":
        model.add(tf.keras.layers.SimpleRNN(50, return_sequences=True, input_shape=(window_size, 1)))
        model.add(tf.keras.layers.SimpleRNN(50))
    else:  # LSTM
        model.add(tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(window_size, 1)))
        model.add(tf.keras.layers.LSTM(50))
    
    model.add(tf.keras.layers.Dense(25))
    model.add(tf.keras.layers.Dense(1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

try:
    df = load_data()
    close_data = df[["Close"]]
    scaler = MinMaxScaler()
    scaled_close = scaler.fit_transform(close_data)
    WINDOW_SIZE = 60
    last_sequence = scaled_close[-WINDOW_SIZE:]
    
    # Show recent data
    st.subheader("Recent Tesla Stock Prices")
    st.line_chart(df['Close'].tail(30))
    
    # Build model
    with st.spinner(f"Building {model_choice} model..."):
        model = build_model(model_choice, WINDOW_SIZE)
    
    def predict_future(model, seq, days):
        preds = []
        current_seq = seq.copy()
        for _ in range(days):
            pred = model.predict(current_seq.reshape(1, WINDOW_SIZE, 1), verbose=0)[0][0]
            preds.append(pred)
            current_seq = np.append(current_seq[1:], pred)
        return scaler.inverse_transform(np.array(preds).reshape(-1, 1))
    
    if st.button("🔮 Predict", type="primary"):
        with st.spinner("Generating predictions..."):
            predictions = predict_future(model, last_sequence, days)
        
        st.success(f"✅ Predictions generated for next {days} day(s)!")
        
        cols = st.columns(len(predictions))
        for i, col in enumerate(cols):
            col.metric(f"Day {i+1}", f"${predictions[i][0]:.2f}")
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(range(1, days+1), predictions, marker="o", linewidth=2, markersize=8)
        ax.set_title(f"{model_choice} – {days} Day Forecast", fontsize=14, fontweight='bold')
        ax.set_xlabel("Days Ahead")
        ax.set_ylabel("Predicted Price ($)")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        st.info("💡 Note: These are untrained model predictions for demonstration. For accurate results, train the models with historical data.")

except FileNotFoundError:
    st.error("❌ TSLA.csv file not found. Please ensure the dataset is uploaded.")
except Exception as e:
    st.error(f"❌ Error: {str(e)}")
    st.info("This is a demo app. The model generates random predictions for demonstration purposes.")
