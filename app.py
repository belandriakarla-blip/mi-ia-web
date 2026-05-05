import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from streamlit_drawable_canvas import st_canvas

# Configuración de la página
st.title("Mi Clasificador de Números IA")
st.write("Dibuja un número del 0 al 9 en el cuadro de abajo.")

# Cargar el modelo que descargaste
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model('modelo_mnist.keras')

model = load_my_model()

# Crear el lienzo para dibujar
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
    # Procesar la imagen dibujada
    img = cv2.cvtColor(canvas_result.image_data.astype('uint8'), cv2.COLOR_RGBA2GRAY)
    img = cv2.resize(img, (28, 28))
    img = img / 255.0
    img = np.expand_dims(img, axis=(0, -1))

    if st.button('Predecir'):
        prediction = model.predict(img)
        confianza = np.max(prediction)
        clase = np.argmax(prediction)

        if confianza < 0.80:
            st.warning(f"No estoy muy segura. Confianza: {confianza:.2f}. ¡Intenta dibujar más claro!")
        else:
            st.success(f"¡Es un {clase}! (Confianza: {confianza:.2f})")