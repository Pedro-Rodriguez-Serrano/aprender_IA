"""
K-Means es un algoritmo de Machine Learning no supervisado.

La diferencia principal con lo que has visto hasta ahora es que
no tenemos una variable y que indique la respuesta correcta.

Clustering = agrupar datos similares.

Por ejemplo, tenemos clientes:

Edad	Salario
20	    20.000
22	    22.000
25	    25.000
45	    60.000
48	    65.000
50	    70.000

K-Means podría descubrir:

GRUPO 1              GRUPO 2

20 años              45 años
22 años              48 años
25 años              50 años
20-25k €             60-70k €

Sin que nosotros le hayamos dicho previamente cuáles son los grupos.

En K-Means:

K = número de grupos


¿Cómo funciona K-Means?

Paso 1

Elegimos cuántos grupos queremos:

K = 2

Paso 2

Coloca 2 puntos llamados centroides.

       ●

                 ●
Paso 3

Cada cliente se asigna al centroide más cercano.

● ● ●        ● ● ●
     ↓      ↓
     C1     C2
Paso 4

Calcula el centro de cada grupo.

Paso 5

Mueve los centroides.

Paso 6

Repite hasta que los grupos se estabilizan.

"""

#Ejemplo:

import pandas as pd

data = {
    "edad": [20, 22, 25, 24, 27,
             45, 48, 50, 52, 47],

    "salario": [20000, 22000, 25000, 23000, 27000,
                60000, 65000, 70000, 75000, 62000]
}

df = pd.DataFrame(data)

print(df)

# Hacer X
X = df[["edad", "salario"]]

#Escalar los datos
#K-Means utiliza distancias, por lo que el escalado es importante.

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

#Crear K-Means

from sklearn.cluster import KMeans

modelo = KMeans(
    n_clusters=2,
    random_state=42,
    n_init=10
)

"""
n_clusters=2

significa: Quiero encontrar 2 grupos.
"""

#Entrenar
modelo.fit(X_scaled)

#Obtener los grupos
grupos = modelo.labels_

print("Los grupos son: \n",grupos)

"""
Recibimos:
 [0 0 0 0 0 1 1 1 1 1]

Lo que significa:

Clientes 1-5 → Grupo 1

Clientes 6-10 → Grupo 0

El 0 y el 1 no significan "malo" y "bueno", ni "no compra" y "compra".

Son simplemente identificadores de grupos.
"""

#Añadir el grupo al DataFrame
df["grupo"] = modelo.labels_

print(df)

"""
Resultado:

   edad  salario  grupo
0    20    20000      0
1    22    22000      0
2    25    25000      0
3    24    23000      0
4    27    27000      0
5    45    60000      1
6    48    65000      1
7    50    70000      1
8    52    75000      1
9    47    62000      1

Grupo 0 → clientes mayores con salarios altos

Grupo 1 → clientes jóvenes con salarios bajos
"""

"""
Cada grupo tiene un centroide.

Podemos obtenerlos con:
"""

print("Centroides escalados: \n",modelo.cluster_centers_)

# Si queremos obtener centroides sin escalar:

centroides = scaler.inverse_transform(
    modelo.cluster_centers_
)

print("Centroides desescalados: ",centroides)

#El centroide representa aproximadamente el "cliente medio" de cada grupo.

"""
Una parte importante del k-means es descrubrir la K

Una técnica muy utilizada es el método del codo (Elbow Method).

Probamos diferentes valores:
"""

for k in range(1, 10):

    modelo = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    modelo.fit(X_scaled)

    print(k, modelo.inertia_)

"""
inertia_ mide, simplificando, qué tan cerca están los puntos de sus centroides.

Normalmente buscamos un punto donde añadir más grupos deja de mejorar mucho el resultado.

Ese punto parece un codo.

El resultado:
1 20.000000000000004
2 0.7375117791980419
3 0.337971143777292
4 0.1567923432902697
5 0.1136747099058442
6 0.0750701192148026
7 0.03650200612447822
8 0.01980479224055265
9 0.007298717643324086

Lo importante es que no eliges la K que tenga la inertia más baja. 
Si hicieramos eso, siempre elegiríamos K=9, porque al aumentar el número
de grupos la inercia siempre tiende a disminuir.

Lo que buscas es el "codo", es decir, el punto a partir del cual añadir más grupos
aporta una mejora mucho menor.

El cambio brutal ocurre entre:

K=1 → 20.00
K=2 → 0.74

Pasas de 20 a 0.74. Es una mejora enorme.

Después:

K=2 → 0.74
K=3 → 0.34
K=4 → 0.16
K=5 → 0.11
...

Las mejoras empiezan a ser mucho menores.

Por tanto, en tus datos K=2 parece una elección muy razonable.

Una forma visual de entenderlo


Inertia
 20 |●
    |
    |
 10 |
    |
  1 |    ●
    |       ●
    |          ●
  0 |             ● ● ● ● ●
    +--------------------------
       1  2  3  4  5  6  7  8  9
                    K

También se puede utilizar:
"""

from sklearn.metrics import silhouette_score

for k in range(2, 10):

    modelo = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = modelo.fit_predict(X_scaled)

    score = silhouette_score(X_scaled, labels)

    print(k, score)

"""
comprueba qué tan buenos/separados son los grupos.
"""
