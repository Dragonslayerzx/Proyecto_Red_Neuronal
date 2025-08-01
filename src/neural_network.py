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
        #Propagacion hacia adelante
        self.activaciones = [X]

        for i, w in enumerate(self.pesos):
            z = np.dot(self.activaciones[-1], w)
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
        #Evaluación de precisión
        predicciones = self.forward(X_test) > 0.5
        print("Predicciones:", predicciones.astype(int).flatten())
        print("Etiquetas reales:", y_test.astype(int).flatten())
        precision = np.mean(predicciones == y_test.reshape(-1, 1))
        y_test = y_test.reshape(-1, 1)
        print(f"Precisión en test: {precision * 100:.2f}%")

    # 3. Matriz de confusión manual
        TP = np.sum((predicciones == 1) & (y_test == 1))
        FP = np.sum((predicciones == 1) & (y_test == 0))
        TN = np.sum((predicciones == 0) & (y_test == 0))
        FN = np.sum((predicciones == 0) & (y_test == 1))
    
        print("\nMatriz de Confusión:")
        print(f"                Predicción 0   Predicción 1")
        print(f"Real 0 (Ausente)    {TN:5}           {FP:5}")
        print(f"Real 1 (Presente)   {FN:5}           {TP:5}")
    
    # 4. Métricas por clase
        if TP + FP > 0:
            precision = TP / (TP + FP)
            recall = TP / (TP + FN)
            print(f"\nClase 1 (Rostro presente):")
            print(f" - Precisión: {precision:.2%}")
            print(f" - Recall:    {recall:.2%}")
    
        if TN + FN > 0:
            specificity = TN / (TN + FP)
            print(f"\nClase 0 (Rostro ausente):")
            print(f" - Especificidad: {specificity:.2%}")
    
    # 5. Verificación de posibles problemas
        if FP == 0 and FN == 0:
            print("\n⚠️ ¡Precisión del 100% detectada! Verifica:")
            print("- ¿El conjunto de test es muy pequeño?")
            print("- ¿Hay contaminación entre train y test?")
            print("- ¿Las imágenes son demasiado simples o uniformes?")