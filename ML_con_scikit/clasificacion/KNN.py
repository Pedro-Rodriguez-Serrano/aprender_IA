"""

KNN (K-Nearest Neighbors) es un algoritmo de clasificación en el que
un dato nuevo se clasifica según la clase de sus vecinos más cercanos.

Ejemplo con este dataset:

| Edad | Compra |
| ---: | ------ |
|   20 | No     |
|   22 | No     |
|   25 | No     |
|   35 | Sí     |
|   40 | Sí     |
|   45 | Sí     |

"""

import pandas as pd

data = {
    "edad":[20,22,25,35,40,45],
    "compra":[False,False,False,True,True,True]}

df = pd.DataFrame(data)

print("Dataset \n",df)


#Separar X e y (datos que queremos usar para predecir compra)
X = df[["edad"]]

y = df["compra"]

#Separar entrenamiento y prueba
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)

# Escalar las variables (Muy importante porque knn usa distancias)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

# Crear modelo KNN
from sklearn.neighbors import KNeighborsClassifier

modelo = KNeighborsClassifier(
    n_neighbors=3
)

"""
n_neighbors=3 significa que con un cliente nuevo mirará los 3 mas cercanos
"""

#Entrenar:
modelo.fit(X_train, y_train)

# Hacer predicciones
predicciones = modelo.predict(X_test)

print("Predicción \n",predicciones)
# Predice True, True (Compra, Compra)

# Evaluar el modelo
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(
    y_test,
    predicciones
)

print("Accuracy:", accuracy) # Si acurracy es 1.0 = 100% de aciertos

# Matriz de confusion:
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(
    y_test,
    predicciones
)

print("Matriz confusión \n",cm)

#Predecir un cliente nuevo

nuevo_cliente = [[36]]

nuevo_cliente = scaler.transform(nuevo_cliente)

prediccion = modelo.predict(nuevo_cliente)

print("Predicción \n", prediccion)

"""
¿Qué ha hecho KNN realmente?

Si entra cliente Edad = 36

KNN busca los 3 clientes más cercanos.

Podría encontrar (Los mas cercanos:):

Cliente A → 35 años → compra
Cliente B → 38 años → compra
Cliente C → 32 años → no compra

La mayoria compra, por eso intuye que este también

Si cambiamos la k y le ponemos demasiado bajo como k=1 podría hacer overfiting 
y ser demasiado sensible a datos. Si hacer una k demasiado alta podría ser demasiado
amplios los datos y hacer underfiting

¿Cuándo usar KNN?

✔ Dataset pequeño o mediano.

✔ Pocas variables.

✔ Cuando la relación entre las clases no es lineal.

No suele ser la mejor opción para millones de registros o miles de variables.

"""