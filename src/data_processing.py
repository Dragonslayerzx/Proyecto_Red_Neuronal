import numpy as np
import cv2
import os
from config import TEST_SIZE, SHUFFLE_DATA, RANDOM_STATE

def cargar_y_preprocesar_imagenes(ruta_directorio, size):
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
    extensiones_validas = ('.jpg', '.jpeg', '.png')

    #Clase 1: Rostro presente
    ruta_rostro_presente = os.path.join(ruta_directorio, 'rostro_presente')
    for filename in os.listdir(ruta_rostro_presente):
        if not filename.lower().endswith(extensiones_validas):
            continue # Ignorar archivos no válidos
        img_ruta = os.path.join(ruta_rostro_presente, filename)
        img = cv2.imread(img_ruta, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, size)
        imagenes.append(img)
        etiquetas.append(1)  # Etiqueta 1

    #Clase 0: Rostro ausente
    ruta_rostro_ausente = os.path.join(ruta_directorio, 'rostro_ausente')
    for filename in os.listdir(ruta_rostro_ausente):
        if not filename.lower().endswith(extensiones_validas):
            continue # Ignorar archivos no válidos
        img_ruta = os.path.join(ruta_rostro_ausente, filename)
        img = cv2.imread(img_ruta, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, size)
        imagenes.append(img)
        etiquetas.append(0)  # Etiqueta 0

    #Convertir listas a numpy arrrays y normalizar
    X = np.array(imagenes).reshape(-1, size[0] * size[1]) / 255.0
    y = np.array(etiquetas)

    print(f"Imagenes cargadas: {len(imagenes)} (Rostro presente: {sum(etiquetas)}, Rostro ausente: {len(etiquetas) - sum(etiquetas)})")
    return X, y

def dividir_manual(X, y, test_size=TEST_SIZE, shuffle=SHUFFLE_DATA, random_seed=RANDOM_STATE):
    """
    Divide los datos en conjuntos de entrenamiento y prueba.
    
    Args:
        X (np.array): Datos de entrada.
        y (np.array): Etiquetas.
        test_size (float): Proporción del conjunto de prueba.
        shuffle (bool): Si se deben mezclar los datos antes de dividirlos.
        random_state (int): Semilla para la aleatoriedad.
        
    Returns:
        X_train, X_test, y_train, y_test: Conjuntos divididos.
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    n_samples = len(X)
    n_test = int(n_samples * test_size)

    if shuffle:
        indices = np.random.permutation(n_samples)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
    else:
        X_shuffled, y_shuffled = X, y
    
    return X_shuffled[:-n_test], X_shuffled[-n_test:], y_shuffled[:-n_test], y_shuffled[-n_test:] 


def dividir_y_guardar_datos(X, y, ruta_guardado):
    """
    Divide los datos en conjuntos de entrenamiento y prueba, y guarda los conjuntos en archivos .npz.
    
    Args:
        X (np.array): Datos de entrada.
        y (np.array): Etiquetas.
        ruta_guardado (str): Ruta donde guardar los archivos.
    """
    X_train, X_test, y_train, y_test = dividir_manual(X, y)
    
    np.savez(os.path.join(ruta_guardado, 'datos_entrenamiento.npz'), X=X_train, y=y_train)
    np.savez(os.path.join(ruta_guardado, 'datos_prueba.npz'), X=X_test, y=y_test) 
    print(f"Split de datos: (Test: {TEST_SIZE*100}%, Shuffle:{SHUFFLE_DATA}, Semilla: {RANDOM_STATE})")
    print(f"Datos guardados en {ruta_guardado}")