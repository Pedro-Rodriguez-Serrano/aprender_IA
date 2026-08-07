"""
El Support Vector Machine (SVM) es un algoritmo de clasificación (y también de regresión)
cuyo objetivo es separar las clases con la mayor distancia posible entre ellas.

¿Cómo funciona?

Imagina este problema:

Clase A (○)

○   ○   ○


Clase B (×)

×   ×   ×

Hay muchas líneas que pueden separar ambas clases.
Por ejemplo:

○ ○ ○
-------
× × ×

SVM busca la mejor línea, es decir, la que deja el mayor margen entre ambas clases.

Si el margen es grande se evita que con cualquier pequeño movimiento la separación quede en el lado equivocado.

Support Vectors:

Son los puntos que están más cerca de la frontera de decisión.

Por ejemplo:
○ ○ ●
---------
▲ × ×

Los puntos: '●' y '▲' son los Support Vectors, los que determinan dónde se coloca la frontera.
El resto de puntos apenas influye en la posición de esa frontera.

Funciona tan bien porque busca una separación que sea robusta.

No intenta simplemente separar las clases, sino hacerlo dejando el mayor "colchón" posible.

El problema es que a veces no existe una línea que separe las clases.

Ejemplo:

○ ○ ○

× × ×

○ ○ ○

Una línea recta no sirve.

Aquí entra en juego el Kernel.

El kernel permite separar datos que no son separables mediante una línea recta.

Los kernels mas usados son:

Kernels más usados
Linear para datos casi lineales.
RBF (Radial Basis Function) para fronteras con curvas
Polynomial para cuando los datos siguen una forma polinómica.
"""

#Ejemplo practico:

import pandas as pd

data = {
    "edad":[20,22,25,30,35,40,45,50],
    "salario":[18000,22000,25000,35000,45000,55000,65000,75000],
    "compra":[0,0,0,0,1,1,1,1]
}

df = pd.DataFrame(data)

#Variables:
X = df[["edad","salario"]]

y = df["compra"]

#Entrenamiento:
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Escalado (En SVM es muy importante escalar.)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

# Crear modelo

from sklearn.svm import SVC

modelo = SVC()

# por defecto:
kernel="rbf"

#Entrenar:
modelo.fit(X_train, y_train)

#Predecir:
predicciones = modelo.predict(X_test)

#Evaluar
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, predicciones)

print(accuracy)

"""
SVM suele ser la mejor opción:

✅ Cuando el dataset es pequeño o mediano.

✅ Cuando las clases están relativamente bien separadas.

✅ Cuando buscas alta precisión.

❌ No suele ser la mejor opción para millones de registros, 
donde algoritmos como Random Forest, XGBoost o redes neuronales suelen escalar mejor.
"""