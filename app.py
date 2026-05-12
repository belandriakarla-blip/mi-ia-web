import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from streamlit_drawable_canvas import st_canvas

# Título de la web
st.title("IA Clasificadora de Números")

# Cargar el modelo
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model('modelo_mnist.keras')

try:
    model = load_my_model()
    st.success("Modelo cargado correctamente")
except Exception as e:
    st.error(f"Error al cargar el modelo: {e}")

# Lienzo para dibujar
canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 1)",
    stroke_width=20,
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

if canvas_result.image_data is not None:
    if st.button('Predecir'):
        # Procesamiento básico
        img = cv2.cvtColor(canvas_result.image_data.astype('uint8'), cv2.COLOR_RGBA2GRAY)
        img = cv2.resize(img, (28, 28))
        img = img / 255.0
        img = np.expand_dims(img, axis=(0, -1))
        
        prediction = model.predict(img)
        clase = np.argmax(prediction)
        st.write(f"### ¡Es un {clase}!")
