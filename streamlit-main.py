import streamlit as st
import numpy as np
import pyttsx3
from src.inference_models import OCRModel, TextDetectionModel
from src.inference_config import OCR_CONFIG
from PIL import Image
import time

#load model
@st.experimental_singleton
def load_model():
    text_detection_model = TextDetectionModel()
    ocr_model = OCRModel(text_detection_model)
    return text_detection_model, ocr_model
@st.experimental_singleton
def load_tts_engine():
    return pyttsx3.init()
text_detection_model, ocr_model = load_model()
engine = load_tts_engine()
#take image
img_file_buffer = st.camera_input("Take a picture")
if img_file_buffer is not None:
    # To read image file buffer as a PIL Image:
    img = Image.open(img_file_buffer)
    # To convert PIL Image to numpy array:
    img = np.array(img)
    #get text
    text = []
    if OCR_CONFIG['model_type']=='CRNN':
        text.append(ocr_model.ocr_with_crnn(img))
    else:
        text.append(ocr_model.ocr_with_tesseract(img))
    #tts
    print(text)
    if not text:
        text = ""
    else:
        text = text[0]
    start = time.time()
    engine.save_to_file(text, 'audio/audio.mp3')
    engine.runAndWait()
    audio_file = open('audio/audio.mp3', 'rb')
    audio_bytes = audio_file.read()
    end = time.time()
    print(f'[INFO] from TTS took {end-start} seconds')
    st.audio(audio_bytes)
    st.text(text)