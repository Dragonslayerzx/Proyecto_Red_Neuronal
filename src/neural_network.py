import os
import numpy as np

class RedNeuronal:
    def __init__(self, capas):
        """
        Inicializa la red neuronal con las capas especificadas.
        Args:
            capas (list): Lista con el número de neuronas por capa. Ej: [480000, 64, 1] para entrada de 800x600.
        """
        self.pesos = []
        for i in range(len(capas) - 1):
            # Inicializacion He para ReLU
            limite = np.sqrt(2 / capas[i])
            self.pesos.append(np.random.randn(capas[i], capas[i + 1]) * limite)

    def relu(self, x):
        return np.maximum(0, x)
    
    def sigmoide(self, x):
        return 1 / (1 + np.exp(-x))
    
    def forward(self, X):
        "Propagacion hacia adelante"
        self.activaciones = [X]

        for i, w in enumerate(self.pesos):
            z = np.dot(self.activaciones[-1], w)
            a = self.relu(z) if i < len(self.pesos) - 1 else self.sigmoide(z)
            self.activaciones.append(a)
        return self.activaciones[-1]
    
    def entrenar(self, X, y, epochs = 1000, lr = 0.01):
        """
        Entrena la red neuronal usando descenso de gradiente "backpropagation".
        
        Args:
            X: Datos de entrada (matriz de caracteristicas).
            y: Etiquetas (vector objetivo).
            epochs: Numero de iteraciones de entrenamiento.
            lr: Tasa de aprendizaje.
        """
        for epoch in range(epochs):
            # Forward pass (guarda activaciones en self.activaciones)
            output = self.forward(X)

            #Backpropagation
            error = output - y.reshape(-1, 1)
            deltas = [error * output * (1 - output)]  # Delta capa salida

            #Propagacion hacia atras
            for i in reversed(range(len(self.pesos) - 1)):
                #Delta para ReLU capas ocultas
                deltas.insert(0, np.dot(deltas[0], self.pesos[i + 1].T) * (self.activaciones[i + 1] > 0))

            #Actualizacion de pesos (todas las capas)
            for i in range(len(self.pesos)):
                self.pesos[i] -= lr * np.dot(self.activaciones[i].T, deltas[i])

            if epoch % 100 == 0:
                loss = np.mean(error ** 2)
                print(f"Epoca {epoch}: Loss = {loss:.6f}")
        
    def evaluar(self, X_test, y_test):
        """Evaluación de precisión"""
        predicciones = self.forward(X_test) > 0.5
        precision = np.mean(predicciones == y_test.reshape(-1, 1))
        print(f"Precisión en test: {precision * 100:.2f}%")


#Bloque principal para ejecutar el entrenamiento
if __name__ == "__main__":
    # 1. Cargar datos usando rutas absolutas basadas en la ubicación de este archivo
    base_dir = os.path.dirname(os.path.abspath(__file__))
    datos_train = np.load(os.path.join(base_dir, "../data/data_processed/datos_entrenamiento.npz"))
    datos_test = np.load(os.path.join(base_dir, "../data/data_processed/datos_prueba.npz"))
    print(datos_train['X'].shape)  # Esto te dará (num_muestras, tamaño_vector)
    # 2. Crear modelo
    input_size = datos_train['X'].shape[1]  # Tamaño de entrada basado en los datos
    modelo = RedNeuronal(capas=[input_size, 64, 1])  # 800x600 píxeles → 480000 neuronas entrada
    
    # 3. Entrenar
    modelo.entrenar(
        X=datos_train['X'],
        y=datos_train['y'],
        epochs=1000,
        lr=0.0001   #Tasa de aprendizaje ajustada
    )
    
    # 4. Evaluar
    modelo.evaluar(
        X_test=datos_test['X'],
        y_test=datos_test['y']
    )