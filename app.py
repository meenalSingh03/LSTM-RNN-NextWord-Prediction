import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

# Function to create a custom LSTM layer without deprecated parameters
def create_custom_lstm(**kwargs):
    """Remove deprecated parameters from LSTM layer"""
    # Remove deprecated parameters that cause compatibility issues
    kwargs.pop('time_major', None)
    kwargs.pop('constants', None)
    
    from tensorflow.keras.layers import LSTM
    return LSTM(**kwargs)

# Function to load model with compatibility fixes
@st.cache_resource
def load_lstm_model():
    """Load the LSTM model with multiple fallback strategies"""
    
    try:
        # Strategy 1: Try loading with custom objects to handle deprecated parameters
        st.info("🔄 Loading model (Strategy 1: Custom objects)...")
        custom_objects = {
            'LSTM': create_custom_lstm
        }
        model = load_model('next_word_lstm.h5', custom_objects=custom_objects)
        st.success("✅ Model loaded successfully with custom objects!")
        return model
        
    except Exception as e1:
        st.warning(f"Strategy 1 failed: {str(e1)}")
        
        try:
            # Strategy 2: Try loading with compile=False
            st.info("🔄 Loading model (Strategy 2: No compilation)...")
            model = load_model('next_word_lstm.h5', compile=False)
            st.success("✅ Model loaded successfully without compilation!")
            return model
            
        except Exception as e2:
            st.warning(f"Strategy 2 failed: {str(e2)}")
            
            try:
                # Strategy 3: Manual fix - create new model and load weights
                st.info("🔄 Loading model (Strategy 3: Weights only)...")
                
                # First, we need to get the tokenizer to determine vocab size
                with open('tokenizer.pickle','rb') as handle:
                    temp_tokenizer = pickle.load(handle)
                
                vocab_size = len(temp_tokenizer.word_index) + 1
                
                # Create a compatible model architecture
                from tensorflow.keras.models import Sequential
                from tensorflow.keras.layers import Embedding, LSTM, Dense
                
                # Estimate parameters (you might need to adjust these based on your original model)
                embedding_dim = 100
                lstm_units = 150
                max_sequence_len = 100  # Default value, will be updated later
                
                new_model = Sequential([
                    Embedding(vocab_size, embedding_dim, input_length=max_sequence_len-1),
                    LSTM(lstm_units),
                    Dense(vocab_size, activation='softmax')
                ])
                
                # Try to load weights from the h5 file
                try:
                    new_model.load_weights('next_word_lstm.h5')
                    st.success("✅ Model architecture recreated and weights loaded!")
                    return new_model
                except:
                    st.error("❌ Could not load weights. The model file might be incompatible.")
                    return None
                    
            except Exception as e3:
                st.error(f"Strategy 3 failed: {str(e3)}")
                
                # Strategy 4: Create a dummy model for demonstration
                st.error("❌ All strategies failed. Please check the model file compatibility.")
                st.info("💡 Suggestions:")
                st.write("1. Try downgrading TensorFlow: `pip install tensorflow==2.15.0`")
                st.write("2. Retrain your model with current TensorFlow version")
                st.write("3. Convert your model to SavedModel format instead of .h5")
                return None

# Load the LSTM Model with error handling
model = load_lstm_model()

# Load the tokenizer
@st.cache_resource
def load_tokenizer():
    """Load the tokenizer with error handling"""
    try:
        with open('tokenizer.pickle','rb') as handle:
            tokenizer = pickle.load(handle)
        st.success("✅ Tokenizer loaded successfully!")
        return tokenizer
    except FileNotFoundError:
        st.error("❌ tokenizer.pickle file not found!")
        return None
    except Exception as e:
        st.error(f"❌ Error loading tokenizer: {e}")
        return None

tokenizer = load_tokenizer()

# Function to predict the next word
def predict_next_word(model, tokenizer, text, max_sequence_len):
    """Predict the next word with improved error handling"""
    try:
        token_list = tokenizer.texts_to_sequences([text])[0]
        
        if not token_list:  # Handle case where no words are recognized
            return "Unknown word(s) in input"
            
        if len(token_list) >= max_sequence_len:
            token_list = token_list[-(max_sequence_len-1):]  # Ensure the sequence length matches max_sequence_len-1
            
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = model.predict(token_list, verbose=0)
        predicted_word_index = np.argmax(predicted, axis=1)[0]  # Fixed indexing issue
        
        for word, index in tokenizer.word_index.items():
            if index == predicted_word_index:
                return word
        return "Unknown"
        
    except Exception as e:
        return f"Prediction error: {str(e)}"

# Function to get top N predictions
def predict_top_words(model, tokenizer, text, max_sequence_len, top_n=5):
    """Get top N word predictions with probabilities"""
    try:
        token_list = tokenizer.texts_to_sequences([text])[0]
        
        if not token_list:
            return []
            
        if len(token_list) >= max_sequence_len:
            token_list = token_list[-(max_sequence_len-1):]
            
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted_probs = model.predict(token_list, verbose=0)[0]
        
        # Get top N indices
        top_indices = np.argsort(predicted_probs)[-top_n:][::-1]
        
        # Create reverse mapping from index to word
        index_to_word = {index: word for word, index in tokenizer.word_index.items()}
        
        results = []
        for idx in top_indices:
            if idx in index_to_word:
                word = index_to_word[idx]
                probability = predicted_probs[idx] * 100
                results.append((word, probability))
                
        return results
        
    except Exception as e:
        return [("Error", 0.0)]

# Streamlit app
st.title("🔮 Next Word Prediction With LSTM And Early Stopping")

# Add some styling
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.prediction-box {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# Check if both model and tokenizer are loaded
if model is not None and tokenizer is not None:
    
    # Input section
    st.subheader("📝 Enter Your Text")
    input_text = st.text_input(
        "Enter the sequence of words:",
        value="To be or not to",
        help="Enter a sequence of words and the model will predict the next word"
    )
    
    # Prediction section
    col1, col2 = st.columns([1, 1])
    
    with col1:
        predict_button = st.button("🔮 Predict Next Word", type="primary")
    
    with col2:
        show_top_predictions = st.checkbox("Show top 5 predictions", value=True)
    
    if predict_button and input_text.strip():
        
        # Get max sequence length from model
        try:
            max_sequence_len = model.input_shape[1] + 1
        except:
            max_sequence_len = 100  # Default fallback
        
        with st.spinner("🤔 Thinking..."):
            
            if show_top_predictions:
                # Show top predictions
                top_predictions = predict_top_words(model, tokenizer, input_text, max_sequence_len, top_n=5)
                
                if top_predictions and top_predictions[0][0] != "Error":
                    st.success(f"🎯 **Top prediction: {top_predictions[0][0]}**")
                    
                    st.subheader("📊 Top 5 Predictions:")
                    for i, (word, prob) in enumerate(top_predictions, 1):
                        st.write(f"{i}. **{word}** ({prob:.2f}%)")
                else:
                    st.error("❌ Could not generate predictions")
                    
            else:
                # Show single prediction
                next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
                
                if next_word and not next_word.startswith("Prediction error"):
                    st.success(f"🎯 **Next word: {next_word}**")
                    
                    # Show the complete sentence
                    st.info(f"💭 Complete text: **{input_text} {next_word}**")
                else:
                    st.error(f"❌ {next_word}")
    
    elif predict_button:
        st.warning("⚠️ Please enter some text before predicting!")
        
    # Add some information about the model
    with st.expander("ℹ️ Model Information"):
        try:
            if hasattr(model, 'input_shape'):
                st.write(f"**Model input shape:** {model.input_shape}")
            if tokenizer:
                vocab_size = len(tokenizer.word_index) + 1
                st.write(f"**Vocabulary size:** {vocab_size:,} words")
                st.write(f"**Most common words:** {list(tokenizer.word_index.keys())[:10]}")
        except:
            st.write("Model information not available")

else:
    st.error("❌ Cannot start the application. Please check if both 'next_word_lstm.h5' and 'tokenizer.pickle' files are present and compatible.")
    
    with st.expander("🔧 Troubleshooting"):
        st.write("""
        **If you're seeing compatibility errors:**
        
        1. **Quick fix:** Try downgrading TensorFlow:
           ```bash
           pip install tensorflow==2.15.0
           ```
        
        2. **Long-term fix:** Retrain your model with the current TensorFlow version
        
        3. **Alternative:** Convert your model to SavedModel format:
           ```python
           # In your training script:
           model.save('my_model')  # Instead of model.save('model.h5')
           ```
        
        4. **Check files:** Ensure both files exist in the same directory as this app
        """)

# Footer
st.markdown("---")
st.markdown("*Made with ❤️ using Streamlit and TensorFlow*")
