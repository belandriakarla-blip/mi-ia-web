import streamlit as st
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas
import cv2
import numpy as np

# Configuración de la página [cite: 46, 47]
st.set_page_config(page_title="IA de Karla - Reconocedor de Dígitos")
st.title("Reconocedor de Dígitos en Tiempo Real")
st.write("Dibuja un número del 0 al 9 en el recuadro negro.") [cite: 48]

# 1. Cargar el modelo guardado [cite: 49]
@st.cache_resource [cite: 50]
def load_my_model():
    # El archivo debe llamarse exactamente así en tu GitHub 
    return tf.keras.models.load_model('modelo_mnist.keras')

try:
    model = load_my_model()
    st.sidebar.success("Modelo cargado correctamente")
except Exception as e:
    st.sidebar.error(f"Error al cargar el modelo: {e}")

# 2. Crear el lienzo (Canvas) para dibujar [cite: 54, 55]
canvas_result = st_canvas(
    fill_color="white", 
    stroke_width=20,
    stroke_color="white",
    background_color="black", 
    height=280, 
    width=280,
    drawing_mode="freedraw", 
    key="canvas",
) [cite: 56, 57, 58, 59]

# 3. Procesar el dibujo y predecir [cite: 60]
if canvas_result.image_data is not None:
    # Solo predecir si el usuario ha dibujado algo (botón para controlar la ejecución)
    if st.button('Predecir número'):
        # Convertir a escala de grises y redimensionar a 28x28 [cite: 62, 63, 65]
        img = cv2.cvtColor(canvas_result.image_data.astype('uint8'), cv2.COLOR_RGBA2GRAY)
        img = cv2.resize(img, (28, 28))
        img = img / 255.0  # Normalizar [cite: 65]
        
        # Preparar la imagen para el modelo (1, 28, 28, 1) [cite: 67]
        img_reshape = img.reshape(1, 28, 28, 1)
        
        # Realizar la predicción [cite: 67]
        pred = model.predict(img_reshape)
        clase = np.argmax(pred) [cite: 68]
        confianza = np.max(pred) [cite: 69]
        
        # 4. Mostrar resultados [cite: 70]
        st.subheader(f"Resultado: {clase}")
        if confianza < 0.80: [cite: 71]
            st.warning(f"Confianza baja ({confianza:.2%}). ¿Podrías dibujar más claro?") [cite: 72, 73]
        else:
            st.success(f"Confianza alta: {confianza:.2%}") [cite: 75]
        
        st.bar_chart(pred[0])  # Visualización de probabilidades [cite: 76]
