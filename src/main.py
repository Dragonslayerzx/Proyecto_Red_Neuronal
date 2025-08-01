import os
import numpy as np
from data_processing import cargar_y_preprocesar_imagenes, dividir_y_guardar_datos
from neural_network import RedNeuronal
from predict import predecir_directorio, cargar_modelo
from config import CAPAS, EPOCAS, TASA_APRENDIZAJE, RAW_DIR, PROCESSED_DIR, MODEL_DIR, PREDICT_DIR, IMAGE_SIZE

def entrenar_y_guardar_modelo():
    #Flujo completo de entrenamiento y guardado del modelo
    print("\n============ Entrenamiento del modelo ==============")
    # 0. Mostrar configuración de parametros
    print("\n=== Parámetros de Configuración ===")
    print(f"Arquitectura: {CAPAS}")
    print(f"Épocas: {EPOCAS}")
    print(f"Tasa de aprendizaje: {TASA_APRENDIZAJE}")
    print(f"Tamaño de imagen: {IMAGE_SIZE}")
  
    # 1. Cargar y preprocesar datos
    X, y = cargar_y_preprocesar_imagenes(ruta_directorio=RAW_DIR, size=IMAGE_SIZE)
    dividir_y_guardar_datos(X, y, ruta_guardado=PROCESSED_DIR, test_size=0.2)

    # 2. Crear y entrenar el modelo
    print("\nInicializando el modelo...")
    datos_train = np.load(os.path.join(PROCESSED_DIR, "datos_entrenamiento.npz"))
    modelo = RedNeuronal(capas = CAPAS) #Se ajusta segun necesidad

    print("\nEntrenando Red Neuronal...")
    modelo.entrenar(
        X = datos_train['X'],
        y = datos_train['y'],
        epochs = EPOCAS,  # Número de épocas ajustado
        lr = TASA_APRENDIZAJE  # Tasa de aprendizaje ajustada
    )

    # 3. Evaluar y guardar el modelo
    print("\nEvaluando el modelo...")
    datos_test = np.load(os.path.join(PROCESSED_DIR, "datos_prueba.npz"))
    modelo.evaluar(
        X_test = datos_test['X'], 
        y_test =datos_test['y']
    )

    print("\nGuardando el modelo entrenado...")
    pesos_dict = {f'pesos_{i}': peso for i, peso in enumerate(modelo.pesos)} #Crea los pesos como un diccionario segun la cantidad de capas
    np.savez(MODEL_DIR,
             **pesos_dict  # Ej: {pesos_0: ..., pesos_1: ..., pesos_2: ...}
    )
    print("Modelo guardado exitosamente.")

if __name__ == "__main__":
    print("\n" \
    "======== Red Neuronal de reconocimiento facial =========")

    while True:
        print("\nMENU PRINCIPAL")
        print("1. Entrenar modelo")
        print("2. Predecir imagenes")
        print("3. Salir")

        opcion = input("Seleccione una opcion (1-3): ").strip()

        if opcion == "1":
            entrenar_y_guardar_modelo()
        elif opcion == "2":
            datos = np.load("data/data_processed/modelo_entrenado.npz")  # Ajusta la ruta
            print("Archivos en el modelo:", list(datos.keys()))  # Ej: ['pesos_0', 'pesos_1']
            print("\n=============== Predicción de imágenes =================")
            modelo_dir = MODEL_DIR
            img_dir = PREDICT_DIR

            if not os.path.exists(modelo_dir):
                print("Modelo no encontrado. Por favor, entrene primero.")
                continue
            modelo = cargar_modelo(ruta_modelo=modelo_dir, capas=CAPAS) #Segun modelo entrenado
            predecir_directorio(
                modelo = modelo,
                directorio = img_dir
            )
        elif opcion == "3":
            print("Saliendo del programa...")
            break
        else:
            print("Opción no válida. Por favor, elija una opción válida.")