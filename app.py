import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Simple custom LSTM to handle deprecated parameters
def custom_lstm_loader():
    """Simple function to load model with deprecated parameter handling"""
    from tensorflow.keras.layers import LSTM
    
    # Override LSTM to ignore deprecated parameters
    original_init = LSTM.__init__
    def new_init(self, *args, **kwargs):
        # Remove problematic parameters
        kwargs.pop('time_major', False)
        kwargs.pop('constants', None)
        return original_init(self, *args, **kwargs)
    
    LSTM.__init__ = new_init
    return LSTM

# Try to load the model with simple error handling
@st.cache_resource
def load_lstm_model():
    try:
        # First, patch the LSTM class
        custom_lstm_loader()
        
        # Try to load the model
        model = load_model('next_word_lstm.h5', compile=False)
        return model, "Model loaded successfully!"
    
    except Exception as e:
        return None, f"Error: {str(e)[:100]}..."

# Load tokenizer
@st.cache_resource  
def load_tokenizer():
    try:
        with open('tokenizer.pickle','rb') as handle:
            tokenizer = pickle.load(handle)
        return tokenizer, "Tokenizer loaded successfully!"
    except Exception as e:
        return None, f"Error loading tokenizer: {str(e)[:50]}..."

# Function to predict the next word - simplified
def predict_next_word(model, tokenizer, text, max_sequence_len):
    try:
        token_list = tokenizer.texts_to_sequences([text])[0]
        if len(token_list) >= max_sequence_len:
            token_list = token_list[-(max_sequence_len-1):]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = model.predict(token_list, verbose=0)
        predicted_word_index = np.argmax(predicted, axis=1)[0]
        
        for word, index in tokenizer.word_index.items():
            if index == predicted_word_index:
                return word
        return "Unknown"
    except:
        return "Error in prediction"

# Streamlit app - simplified
st.title("Next Word Prediction With LSTM")

# Load model and tokenizer
model, model_status = load_lstm_model()
tokenizer, tokenizer_status = load_tokenizer()

# Show loading status
if model is None:
    st.error(model_status)
    st.info("💡 Try: pip install tensorflow==2.15.0")
    st.stop()

if tokenizer is None:
    st.error(tokenizer_status)
    st.stop()

st.success("✅ Model and tokenizer loaded!")

# Main interface
input_text = st.text_input("Enter the sequence of Words", "To be or not to")

if st.button("Predict Next Word"):
    if input_text.strip():
        try:
            max_sequence_len = model.input_shape[1] + 1
        except:
            max_sequence_len = 100
            
        next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
        st.write(f'Next word: **{next_word}**')
    else:
        st.warning("Please enter some text")

# Simple troubleshooting section
with st.expander("If you get errors"):
    st.code("pip install tensorflow==2.15.0", language="bash")
    st.write("Then restart the app")
