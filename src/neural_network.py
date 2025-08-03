import os
import numpy as np
import matplotlib.pyplot as plt

class RedNeuronal:
    def __init__(self, capas):
        """
        Inicializa la red neuronal con las capas especificadas.
        Args:
            capas (list): Lista con el número de neuronas por capa. Ej: [480000, 64, 1] para entrada de 800x600.
        """
        self.pesos = []
        self.biases = []
        for i in range(len(capas) - 1):
            # Inicializacion He para ReLU
            limite = np.sqrt(2 / capas[i])
            self.pesos.append(np.random.randn(capas[i], capas[i + 1]) * limite)
            self.biases.append(np.zeros((1, capas[i + 1]))) # Biases inicializados en cero

    def relu(self, x):
        """"
        Función de activación ReLU
        Args:
            x (np.array): Entrada de la función.
        """
        return np.maximum(0, x)
    
    def sigmoide(self, x):
        """
        Función de activación sigmoide
        Args:
            x (np.array): Entrada de la función.
        """
        return 1 / (1 + np.exp(-x))
    
    def forward(self, X):
        """
        Propagación hacia adelante
        Args:
            X (np.array): Datos de entrada.
        """
        self.activaciones = [X]

        for i, w in enumerate(self.pesos):
            z = np.dot(self.activaciones[-1], w) + self.biases[i]
            a = self.relu(z) if i < len(self.pesos) - 1 else self.sigmoide(z)
            self.activaciones.append(a)
        return self.activaciones[-1]
    
    def entrenar(self, X, y, epochs, lr): 
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

            #Cálculo del error y deltas para la capa de salida
            error = output - y.reshape(-1, 1)
            deltas = [error * output * (1 - output)]  # Delta capa salida

            #Backpropagation
            for i in reversed(range(len(self.pesos) - 1)):
                #Delta para ReLU capas ocultas
                deltas.insert(0, np.dot(deltas[0], self.pesos[i + 1].T) * (self.activaciones[i + 1] > 0))

            #Actualizacion de pesos (todas las capas)
            for i in range(len(self.pesos)):
                self.pesos[i] -= lr * np.dot(self.activaciones[i].T, deltas[i]) #Actualiza pesos
                self.biases[i] -= lr * np.sum(deltas[i], axis=0, keepdims=True) #Actualiza bias

            #Impresion de perdida cada 100 epocas
            if epoch % 100 == 0:
                loss = np.mean(error ** 2)
                print(f"Epoca {epoch}: Loss = {loss:.6f}")
        
    def evaluar(self, X_test, y_test):
        """
        Evalúa el modelo en un conjunto de prueba.
        Args:
            X_test (np.array): Datos de entrada para la evaluación.
            y_test (np.array): Etiquetas reales para la evaluación.
        """
        #Forward pass para obtener predicciones
        predicciones = self.forward(X_test) > 0.5
        y_test = y_test.reshape(-1, 1)
        incorrectas = np.where(predicciones != y_test)[0]
        """       
        #Visualizacion de imagenes mal clasificadas (opcional)
        for idx in incorrectas:
            img_flat = X_test[idx]
            img = img_flat.reshape(600, 800)  # Ajusta según IMAGE_SIZE
        
            plt.figure(figsize=(8, 6))
            plt.imshow(img, cmap='gray')
            plt.title(f"Real: {y_test[idx][0]} - Pred: {int(predicciones[idx][0])}")
            plt.colorbar()
            plt.show()
        """
        #Evaluación de precisión
        print(f"\nImágenes mal clasificadas: {len(incorrectas)}/{len(y_test)}")
        print("Predicciones:", predicciones.astype(int).flatten())
        print("Etiquetas reales:", y_test.astype(int).flatten())
        precision = np.mean(predicciones == y_test.reshape(-1, 1))
        print(f"Precisión en test: {precision * 100:.2f}%")

        #Matriz de confusión
        TP = np.sum((predicciones == 1) & (y_test == 1))
        FP = np.sum((predicciones == 1) & (y_test == 0))
        TN = np.sum((predicciones == 0) & (y_test == 0))
        FN = np.sum((predicciones == 0) & (y_test == 1))
    
        print("\n============Matriz de Confusión==============")
        print(f"                Predicción 0   Predicción 1")
        print(f"Real 0 (Ausente)    {TN:5}           {FP:5}")
        print(f"Real 1 (Presente)   {FN:5}           {TP:5}")
    
        #Metricas de evaluación
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

        print("\n=========Metricas de evaluación=========")
        print(f"Clase 1 (Presente):")
        print(f"Precisión: {precision:.2%}  | Recall: {recall:.2%}")
        print(f"\nClase 0 (Ausente):")
        print(f"Especificidad: {specificity:.2%}")