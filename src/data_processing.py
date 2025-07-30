import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split

def cargar_y_preprocesar_imagenes(ruta_directorio, size=(700, 600)):
    """
    Carga y preprocesa imágenes desde un directorio.
    
    Args:
        ruta_directorio (str): Ruta al directorio que contiene las imágenes.
        size (tuple): Tamaño al que redimensionar las imágenes.
        
    Returns:
        X, y: Numpy arrays normalizados.
    """
    imagenes = []
    etiquetas = []

    #Clase 1: Rostro presente
    ruta_rostro_presente = os.path.join(ruta_directorio, 'rostro_presente')
    for filename in os.listdir(ruta_rostro_presente):
        img_ruta = os.path.join(ruta_rostro_presente, filename)
        img = cv2.imread(img_ruta, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, size)
        imagenes.append(img)
        etiquetas.append(1)  # Etiqueta 1

    #Clase 0: Rostro ausente
    ruta_rostro_ausente = os.path.join(ruta_directorio, 'rostro_ausente')
    for filename in os.listdir(ruta_rostro_ausente):
        img_ruta = os.path.join(ruta_rostro_ausente, filename)
        img = cv2.imread(img_ruta, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, size)
        imagenes.append(img)
        etiquetas.append(0)  # Etiqueta 0

    #Convertir listas a numpy arrrays y normalizar
    X = np.array(imagenes).reshape(-1, size[0] * size[1]) / 255.0
    y = np.array(etiquetas)

    print(f"Total de imagenes cargadas: {len(imagenes)} (Rostro presente: {sum(etiquetas)}, Rostro ausente: {len(etiquetas) - sum(etiquetas)})")
    return X, y

def dividir_y_guardar_datos(X, y, ruta_guardado, test_size=0.2, random_state=42):
    """
    Divide los datos en conjuntos de entrenamiento y prueba, y guarda los conjuntos en archivos .npz.
    
    Args:
        X (np.array): Datos de entrada.
        y (np.array): Etiquetas.
        ruta_guardado (str): Ruta donde guardar los archivos.
        test_size (float): Proporción del conjunto de prueba.
        random_state (int): Semilla para la aleatoriedad.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    np.savez(os.path.join(ruta_guardado, 'datos_entrenamiento.npz'), X=X_train, y=y_train)
    np.savez(os.path.join(ruta_guardado, 'datos_prueba.npz'), X=X_test, y=y_test)
    
    print(f"Datos guardados en {ruta_guardado}")

# Funcion principal para ejecutar el procesamiento
if __name__ == "__main__":
    # 1. Definir rutas básicas
    DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
    RAW_DIR = os.path.join(DATA_DIR, "raw_images")
    PROCESSED_DIR = os.path.join(DATA_DIR, "data_processed")
    
    # 2. Procesar y guardar
    X, y = cargar_y_preprocesar_imagenes(RAW_DIR)
    dividir_y_guardar_datos(X, y, PROCESSED_DIR)