"""
2.1 Clasificación

La clasificación es una tarea de aprendizaje supervisado cuyo objetivo
es predecir una categoría o clase a la que pertenece un dato.

Logistic Regression: es un algoritmo de clasificación.

    Sirve para responder preguntas como:

    ¿Comprará el cliente?
    ¿Es spam?
    ¿Aprobará el examen?
    ¿Tiene una enfermedad?

    La salida es una probabilidad entre 0 y 1.

"""

##Ejemplo:

import pandas as pd

data = {
    "edad": [20,22,25,30,35,40,45,50],
    "salario": [18000,22000,25000,35000,45000,55000,65000,75000],
    "compra": [0,0,0,0,1,1,1,1]
}

df = pd.DataFrame(data)

# Paso 1. Separar variables

X = df[["edad","salario"]]

y = df["compra"]

# Siempre:
#   X → variables de entrada
#   y → variable objetivo (target)

# Paso 2. Dividir entrenamiento y prueba

from sklearn.model_selection import train_test_split

# Entrenamiento (train): datos con los que el modelo aprende.
# Prueba (test): datos que el modelo nunca ha visto y que se
#                usan para evaluar qué tan bien funciona.

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

"""
La función devuelve cuatro conjuntos de datos:

X_train: variables de entrada para entrenar.
X_test: variables de entrada para probar.
y_train: respuestas correctas del entrenamiento.
y_test: respuestas correctas de la prueba.

test_size=0.2 significa: 
    80% entrenamiento
    20% prueba
random_state:
    Para obtener siempre la misma división.

El modelo aprende solo con el conjunto de entrenamiento. Después, 
intenta predecir las respuestas del conjunto de prueba (X_test), y 
esas predicciones se comparan con y_test para medir su rendimiento.

"""

# Paso 3. Escalar datos

""" Como Logistic Regression utiliza distancias y optimización, es recomendable escalar.
Si no se hace el algoritmo podría confundir que salario tiene mas peso porque es mayor y
podría tardar mas o que no encuentre la solucion al ser muy diferentes escalas """

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

"""
Nota: Nunca hay que usar fit_transform(X_test) porque estás dejando que el modelo "vea" 
información del conjunto de prueba. Esto se conoce como data leakage (fuga de datos) y
 hace que la evaluación del modelo sea poco fiable.
 
 Una analogía:

Imagina que un profesor prepara un examen.

Entrenamiento: son los apuntes con los que estudias.
Prueba: es el examen.

Si antes de hacer el examen el profesor lo lee y cambia las reglas para adaptarlas a las respuestas de los alumnos, la evaluación deja de ser justa.

Con el escalado pasa lo mismo:

fit() = aprender las reglas (media y desviación).
transform() = aplicar esas reglas.
"""

# Paso 4. Crear el modelo

from sklearn.linear_model import LogisticRegression

modelo = LogisticRegression()

# Paso 5. Entrenar

modelo.fit(X_train, y_train)

"""El modelo busca los parámetros que mejor separan ambas clases."""

# Paso 6. Predecir

predicciones = modelo.predict(X_test)

# Paso 7. Probabilidades

modelo.predict_proba(X_test)

"""
Devuelve algo parecido a:

[[0.92 0.08]
 [0.10 0.90]
 [0.25 0.75]]

Interpretación:

Primera fila:

No compra → 92%

Compra → 8%
"""

# Paso 8. Evaluar

from sklearn.metrics import accuracy_score

accuracy_score(y_test, predicciones)

"""
Ejemplo:

0.90

Significa:

El modelo acierta el 90% de las veces.
"""

# Extra: Matriz de confusión

from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, predicciones)

"""
Permite ver:

Verdaderos positivos
Verdaderos negativos
Falsos positivos
Falsos negativos
"""

# Extra: Classification Report

from sklearn.metrics import classification_report

print(classification_report(y_test, predicciones))

"""
Incluye:

Precision
Recall
F1-score
Accuracy
"""
