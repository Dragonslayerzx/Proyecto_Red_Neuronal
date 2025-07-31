import os
import numpy as np
from data_processing import cargar_y_preprocesar_imagenes, dividir_y_guardar_datos
from neural_network import RedNeuronal
from predict import predecir_directorio, cargar_modelo

def entrenar_y_guardar_modelo():
    #Flujo completo de entrenamiento y guardado del modelo
    print("\n========= Entrenamiento del modelo ===========")

    # 1. Cargar y preprocesar datos
    DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
    X, y = cargar_y_preprocesar_imagenes(os.path.join(DATA_DIR, "raw_images"))
    dividir_y_guardar_datos(X, y, os.path.join(DATA_DIR, "data_processed"))

    # 2. Crear y entrenar el modelo
    print("\nInicializando el modelo...")
    datos_train = np.load(os.path.join(DATA_DIR, "data_processed", "datos_entrenamiento.npz"))
    modelo = RedNeuronal(capas = [800*600, 64, 1]) #Se ajusta segun necesidad

    print("\nEntrenando Red Neuronal...")
    modelo.entrenar(
        X = datos_train['X'],
        y = datos_train['y'],
        epochs = 1000,  # Número de épocas ajustado
        lr = 0.0001  # Tasa de aprendizaje ajustada
    )

    # 3. Evaluar y guardar el modelo
    print("\nEvaluando el modelo...")
    datos_test = np.load(os.path.join(DATA_DIR, "data_processed", "datos_prueba.npz"))
    modelo.evaluar(datos_test['X'], datos_test['y'])

    print("\nGuardando el modelo entrenado...")
    np.savez(os.path.join(DATA_DIR, "data_processed", "modelo_entrenado.npz"),
             pesos_0=modelo.pesos[0], pesos_1=modelo.pesos[1])
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
            DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
            print("\n============= Predicción de imágenes ===============")
            modelo_dir = os.path.join(DATA_DIR, "data_processed", "modelo_entrenado.npz")
            img_dir = os.path.join(DATA_DIR, "predict_images")

            if not os.path.exists(modelo_dir):
                print("Modelo no encontrado. Por favor, entrene primero.")
                continue
            modelo = cargar_modelo(modelo_dir, capas=[800*600, 64, 1]) #Segun modelo entrenado
            predecir_directorio(
                modelo = modelo,
                directorio = img_dir
            )
        elif opcion == "3":
            print("Saliendo del programa...")
            break
        else:
            print("Opción no válida. Por favor, elija una opción válida.")