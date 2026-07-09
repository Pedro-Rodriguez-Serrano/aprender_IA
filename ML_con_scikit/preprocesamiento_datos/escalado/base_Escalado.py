""" El escalado consiste en poner las variables numéricas en una escala comparable
para que ninguna domine a las demás solo por tener números más grandes.
Si por ejemplo tenemos un data set:
| edad | salario |
| ---- | ------- |
| 25   | 30000   |
| 30   | 45000   |
| 35   | 60000   |

Muchos algoritmos pueden interpretar que, como el salario tiene valores mas altos, es mas
importante aunque no sea así.

METODOS PARA ESCALAR:

1. StandardScaler (estandarización):

Transforma los datos para que:

media = 0
desviación estándar = 1

Ejemplo:
"""

edad = [20, 30, 40] # media = 30 / Desviación estándar ≈ 8.16

# transformación:
# 20 → -1.22
# 30 →  0
# 40 →  1.22

# Para hacerlo con código:

#Crear dataframe:
data = {
    "edad": [20, 30, 40]
}

import pandas as pd

df = pd.DataFrame(data)

# Crear scaler y transformar:

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

df["edad"] = scaler.fit_transform(df[["edad"]])

print('df standarizado:\n',df)

"""
Método 2: MinMaxScaler (normalización)

Transforma todos los valores a un rango fijo, normalmente [0,1]

Por ejemplo:
"""

edad = [20, 30, 40] # min 20 max 40

# Tras normalización:
# 20 → 0
# 30 → 0.5
# 40 → 1

from sklearn.preprocessing import MinMaxScaler

data = {
    "edad": [20, 30, 40]
}

df = pd.DataFrame(data)

scaler = MinMaxScaler()

df["edad"] = scaler.fit_transform(df[["edad"]])

print('df normalizado:\n',df)

"""
Los resultados son:

df standarizado:
        edad
0 -1.224745
1  0.000000
2  1.224745
df normalizado:
    edad
0   0.0
1   0.5
2   1.0

Para decidir cual usar hay que ver el contexto.
El StandardScaler se usa mucho en ml clásico, permite conservar bien la distribución 
y es es menos sensible al rango absoluto y outliers.
MinMaxScaler es útil cuando quieres todo entre 0 y 1, es muy usado en redes neuronales, 
pero es muy sensible a outliers.

Ante la duda suele ser mejor StandarScaler porque no limita los valores a un rango fijo y 
no fuerza maximos ni minimos, lo que reduce distorsiones por outliers, la normalización 
fuerza a que todos los valores esten entre 0 y 1 (Max y Min)

"""