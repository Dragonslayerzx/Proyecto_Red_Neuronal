import os

#=================Configuracion de las red neuronal==================
CAPAS = [800*600, 64, 1] # Arquitectura de la red (entrada, ocultas, salida), a[0] debe coincidir con el tamaño de la imagen aplanada
EPOCAS = 1000  # Número de épocas para entrenamiento
TASA_APRENDIZAJE = 0.0001  # Tasa de aprendizaje para el optimizador

#=================Configuracion de rutas==================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
RAW_DIR = os.path.join(DATA_DIR, "raw_images")
PROCESSED_DIR = os.path.join(DATA_DIR, "data_processed")
MODEL_DIR = os.path.join(PROCESSED_DIR, "modelo_entrenado.npz")
PREDICT_DIR = os.path.join(DATA_DIR, "predict_images")

#=================Configuracion de imagenes==================
IMAGE_SIZE = (800, 600)  # Tamaño de las imágenes (ancho, alto)