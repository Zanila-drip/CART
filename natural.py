import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class NodoDecision:
    def __init__(self, caracteristica=None, umbral=None, izquierda=None, derecha=None, valor=None):
        self.caracteristica = caracteristica  # Índice de la característica para dividir
        self.umbral = umbral  # Valor de umbral para la división
        self.izquierda = izquierda  # Subárbol izquierdo (valores <= umbral)
        self.derecha = derecha  # Subárbol derecho (valores > umbral)
        self.valor = valor  # Valor de la hoja (si es un nodo hoja)


def calcular_gini(y):
    clases, conteos = np.unique(y, return_counts=True)
    probabilidades = conteos / len(y)
    gini = 1 - np.sum(probabilidades ** 2)
    return gini


def dividir_dataset(X, y, caracteristica, umbral):
    mascara = X[:, caracteristica] <= umbral
    izquierda_X, izquierda_y = X[mascara], y[mascara]
    derecha_X, derecha_y = X[~mascara], y[~mascara]
    return (izquierda_X, izquierda_y), (derecha_X, derecha_y)


def encontrar_mejor_division(X, y):
    mejor_caracteristica, mejor_umbral, mejor_gini = None, None, float('inf')
    #mejor umbral es la forma para dividir los datos
    #mejor caracteristica proporciona la mayor reduccion de impureza (Gini)
    n_caracteristicas = X.shape[1]

    for caracteristica in range(n_caracteristicas):
        umbrales = np.unique(X[:, caracteristica])
        for umbral in umbrales:
            (izquierda_X, izquierda_y), (derecha_X, derecha_y) = dividir_dataset(X, y, caracteristica, umbral)
            if len(izquierda_y) == 0 or len(derecha_y) == 0:
                continue

            gini_izquierda = calcular_gini(izquierda_y)
            gini_derecha = calcular_gini(derecha_y)
            gini_ponderado = (len(izquierda_y) / len(y)) * gini_izquierda + (len(derecha_y) / len(y)) * gini_derecha

            if gini_ponderado < mejor_gini:
                mejor_caracteristica, mejor_umbral, mejor_gini = caracteristica, umbral, gini_ponderado

    return mejor_caracteristica, mejor_umbral


def construir_arbol(X, y, profundidad=0, max_profundidad=5):
    if profundidad >= max_profundidad or len(np.unique(y)) == 1:
        valor_hoja = np.bincount(y).argmax()  # Clase más común
        return NodoDecision(valor=valor_hoja)

    caracteristica, umbral = encontrar_mejor_division(X, y)
    if caracteristica is None:
        return NodoDecision(valor=np.bincount(y).argmax())

    (izquierda_X, izquierda_y), (derecha_X, derecha_y) = dividir_dataset(X, y, caracteristica, umbral)
    izquierda = construir_arbol(izquierda_X, izquierda_y, profundidad + 1, max_profundidad)
    derecha = construir_arbol(derecha_X, derecha_y, profundidad + 1, max_profundidad)

    return NodoDecision(caracteristica=caracteristica, umbral=umbral, izquierda=izquierda, derecha=derecha)


def predecir_muestra(nodo, x):
    if nodo.valor is not None:
        return nodo.valor
    if x[nodo.caracteristica] <= nodo.umbral:
        return predecir_muestra(nodo.izquierda, x)
    else:
        return predecir_muestra(nodo.derecha, x)


# Cargar dataset de Iris
iris = load_iris()
X = iris.data
y = iris.target

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Construir el árbol
arbol = construir_arbol(X_train, y_train, max_profundidad=3)

# Predecir
y_pred = [predecir_muestra(arbol, x) for x in X_test]

# Evaluar
precision = accuracy_score(y_test, y_pred)
print(f"Precisión: {precision * 100:.2f}%")