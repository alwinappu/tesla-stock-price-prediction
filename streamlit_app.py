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
    df = pd.read_csv("https://raw.githubusercontent.com/alwinappu/tesla-stock-price-prediction/main/TSLA.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    return df

def create_sequences(data, window_size):
    """Create input sequences and targets for training"""
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)

def build_and_train_model(model_type, X_train, y_train, window_size=60, epochs=20):
    """Build and train model with actual data"""
    model = tf.keras.Sequential()
    
    if model_type == "SimpleRNN":
        model.add(tf.keras.layers.SimpleRNN(50, return_sequences=True, input_shape=(window_size, 1)))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.SimpleRNN(50, return_sequences=False))
        model.add(tf.keras.layers.Dropout(0.2))
    else:  # LSTM
        model.add(tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(window_size, 1)))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.LSTM(50, return_sequences=False))
        model.add(tf.keras.layers.Dropout(0.2))
    
    model.add(tf.keras.layers.Dense(25))
    model.add(tf.keras.layers.Dense(1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Train the model
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=0, validation_split=0.1)
    
    return model, history

try:
    df = load_data()
    close_data = df[["Close"]]
    scaler = MinMaxScaler()
    scaled_close = scaler.fit_transform(close_data)
    
    WINDOW_SIZE = 60
    
    # Show recent data
    st.subheader("Recent Tesla Stock Prices")
    st.line_chart(df['Close'].tail(30))
    
    if st.button("🔮 Predict", type="primary"):
        # Create training sequences
        X, y = create_sequences(scaled_close.flatten(), WINDOW_SIZE)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        # Train model
        with st.spinner(f"Training {model_choice} model on {len(X)} samples... This may take 1-2 minutes..."):
            model, history = build_and_train_model(model_choice, X, y, WINDOW_SIZE, epochs=20)
        
        st.success(f"\u2705 Model trained successfully! Final loss: {history.history['loss'][-1]:.6f}")
        
        # Show training progress
        with st.expander("View Training Loss"):
            fig_loss, ax_loss = plt.subplots(figsize=(8, 3))
            ax_loss.plot(history.history['loss'], label='Training Loss')
            ax_loss.plot(history.history['val_loss'], label='Validation Loss')
            ax_loss.set_xlabel('Epoch')
            ax_loss.set_ylabel('Loss')
            ax_loss.set_title('Model Training Progress')
            ax_loss.legend()
            ax_loss.grid(True, alpha=0.3)
            st.pyplot(fig_loss)
        
        # Make predictions
        last_sequence = scaled_close[-WINDOW_SIZE:]
        
        def predict_future(model, seq, days):
            preds = []
            current_seq = seq.copy().flatten()
            for _ in range(days):
                pred = model.predict(current_seq.reshape(1, WINDOW_SIZE, 1), verbose=0)[0][0]
                preds.append(pred)
                current_seq = np.append(current_seq[1:], pred)
            return scaler.inverse_transform(np.array(preds).reshape(-1, 1))
        
        with st.spinner("Generating predictions..."):
            predictions = predict_future(model, last_sequence, days)
        
        st.success(f"\u2705 Predictions generated for next {days} day(s)!")
        
        cols = st.columns(len(predictions))
        for i, col in enumerate(cols):
            col.metric(f"Day {i+1}", f"${predictions[i][0]:.2f}")
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(range(1, days+1), predictions, marker="o", linewidth=2, markersize=8, color='#1f77b4')
        ax.set_title(f"{model_choice} \u2013 {days} Day Forecast", fontsize=14, fontweight='bold')
        ax.set_xlabel("Days Ahead")
        ax.set_ylabel("Predicted Price ($)")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        st.info("💡 Model trained on historical Tesla stock data. Predictions are based on the last 60 days of price movements.")
        
except FileNotFoundError:
    st.error("\u274c TSLA.csv file not found. Please ensure the dataset is uploaded.")
except Exception as e:
    st.error(f"\u274c Error: {str(e)}")
    st.info("An error occurred while processing. Please try again.")
