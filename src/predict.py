import os
import cv2
import numpy as np
from neural_network import RedNeuronal

def cargar_modelo(ruta_modelo, capas):
    """
    Carga un modelo entrenado desde un archivo .npz
    Args:
        ruta_modelo (str): Ruta al archivo .npz con los pesos
        capas (list): Lista con la arquitectura del modelo (ej: [480000, 64, 1])
    Returns:
        RedNeuronal: Modelo con pesos cargados
    """
    datos = np.load(ruta_modelo)
    modelo = RedNeuronal(capas = capas)
    #Se asignan los pesos en orden
    modelo.pesos = [datos[f'pesos_{i}'] for i in range(len(capas) - 1)]
    return modelo

def preprocesar_imagen(ruta_imagen, size=(800, 600)):
    """
    Preprocesa una imagen para que coincida con el formato de entrenamiento
    Args:
        ruta_imagen (str): Ruta a la imagen a predecir
        size (tuple): Tamaño objetivo (ancho, alto)
    Returns:
        np.array: Imagen preprocesada (1, 480000)
    """
    img = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"No se pudo cargar la imagen: {ruta_imagen}")
    img = cv2.resize(img, size) # Redimensionar a 800x600
    return img.reshape(1, -1) / 255.0  # Normalizar y aplanar

def predecir_imagen(modelo, ruta_imagen):
    """
    Realiza una predicción sobre una imagen
    Args:
        modelo (RedNeuronal): Modelo entrenado
        ruta_imagen (str): Ruta a la imagen
    Returns:
        tuple: (predicción, probabilidad)
    """
    X = preprocesar_imagen(ruta_imagen)
    prob = modelo.forward(X)[0][0]  # Probabilidad entre 0 y 1
    return ("TÚ", prob) if prob > 0.5 else ("OTRO", 1 - prob)

def predecir_directorio(modelo, directorio):
    """
    Predice todas las imágenes válidas en un directorio
    Args:
        modelo (RedNeuronal): Modelo entrenado
        directorio (str): Ruta al directorio con imágenes
    """
    print("\n ========= Resultados de las predicciones =========")
    for filename in os.listdir(directorio):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            try:
                ruta_completa = os.path.join(directorio, filename)
                etiqueta, prob = predecir_imagen(modelo, ruta_completa)
                print(f"{filename}: {etiqueta} (Probabilidad: {prob:.2f})")
            except Exception as e:
                print(f"Error procesando {filename}: {e}")

if __name__ == "__main__":
    #Configuracion de rutas
    DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
    MODELO_DIR = os.path.join(DATA_DIR, "data_processed", "modelo_entrenado.npz")
    PREDICT_DIR = os.path.join(DATA_DIR, "predict_images")

    #Carga del modelo
    modelo = cargar_modelo(ruta_modelo = MODELO_DIR, capas = [800*600, 64, 1])

    #Predicciones en el directorio especificado
    predecir_directorio(modelo, PREDICT_DIR)
            