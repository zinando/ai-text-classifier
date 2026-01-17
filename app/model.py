import os
from dotenv import load_dotenv

load_dotenv(override=True)
ENV = os.getenv("ENV", "development").lower()

def model_placeholder(input):
    result = 0.95 if 'good' in input else 0.45
    return result

def load_model(model_path:str):
    """loads AI model from model_path"""
    if ENV == 'production':
        import tensorflow as tf
        model = tf.keras.models.load_model(model_path)
        model.trainable = False
        return model
    return model_placeholder
    
def process_input(data:str):
    """Processes request data defore feeding AI model"""
    return data

def process_output(prediction:float):
    """Process output based on prediction result"""
    return "positive" if prediction > 0.5 else "negative"

def predict_text(model, text: str):
    pre_processed = process_input(text)
    confidence = model(pre_processed)
    return process_output(confidence), confidence
