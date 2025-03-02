"""
Algoritmo CART Utilizando librerias
codigo rescatado de: https://github.com/TrainingByPackt/Applied-Supervised-Learning-with-Python/blob/master/Chapter%204%20-%20Classification/Exercise%2042%20-%20Iris%20Classification%20Using%20a%20CART%20Decision%20Tree.ipynb?short_path=fd4eea2
ultima modificacion: 25/02/2025
participantes: Manzanilla Martinez Leonardo Manuel || OLIVARES CONTRERAS ALEJANDRO
"""


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

# Cargar el conjunto de datos Iris
iris = load_iris()
dataset = pd.DataFrame(data=iris.data, columns=iris.feature_names)
print(dataset.head(14))
dataset['especie'] = iris.target  # Añadir la columna de la especie

# Conjunto de características (X) y etiquetas (y)
X = dataset.iloc[:, 0:4].values  # Características: longitud y ancho de sépalos y pétalos
y = dataset.iloc[:, 4].values    # Etiquetas: especie de la flor

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el modelo
modelo = DecisionTreeClassifier(criterion='gini', random_state=42)
modelo.fit(X_train, y_train)

# Predecir en el conjunto de prueba
y_pred = modelo.predict(X_test)

# Evaluar el modelo
precision = accuracy_score(y_test, y_pred)
print(f"\nPrecisión del modelo: {precision:.2f}")

# Ejemplo de predicción
ejemplo = [[5.1, 3.5, 1.4, 0.2]]  # Datos de ejemplo (longitud y ancho de sépalos y pétalos)
prediccion = modelo.predict(ejemplo)

