import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Cargar el conjunto de datos
dataset = pd.read_csv('temp.csv')
dataset = pd.DataFrame(data=dataset.iloc[:, 1:6].values, columns=["outlook", "temprature", "humdity", "windy", "play"])

# Codificar las características
dataset_codificado = dataset.iloc[:, 0:5]
le = LabelEncoder()

for i in dataset_codificado.columns:
    dataset_codificado[i] = le.fit_transform(dataset_codificado[i])

print(dataset)  # Conjunto de datos original para comparación

# Conjunto de características (X)
X = dataset_codificado.iloc[:, 0:4].values
# Conjunto de etiquetas (y)
y = dataset_codificado.iloc[:, 4].values

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el modelo
modelo = DecisionTreeClassifier(criterion='gini')
modelo.fit(X_train, y_train)

# Predecir en el conjunto de prueba
y_pred = modelo.predict(X_test)

# Evaluar el modelo
precision = accuracy_score(y_test, y_pred)
print(f"Precisión: {precision:.2f}")

# Ejemplo de predicción
if modelo.predict([[0, 1, 0, 1]]) == 1:
    print("Sí, puedes jugar")
else:
    print("No, no puedes jugar")