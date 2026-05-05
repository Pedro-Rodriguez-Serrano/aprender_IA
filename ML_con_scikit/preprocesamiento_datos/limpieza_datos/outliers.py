import pandas as pd
import numpy as np

""" *Los outliers (o valores atípicos) son datos
# que se alejan mucho del resto de observaciones dentro de un conjunto de datos.
# Por ejemplo, en este dataset: [20, 22, 21, 19, 23, 20, 22, 21, 150]
# el oulier es 150 porque se aleja mucho de la media.

# El outlier se distingue por el contexto, si el dataset es de edades, lo mas probable
# es que un valos de 150 sea falso, si son temperaturas, tambien porque es fisicamente
# imposible que un termometro en la tierra haya registrado datos de 150º. Si los datos
# son transacciones económicas, tiene sentido.
# Si hay muchos valores extremos puede que los datos extremos no lo sean, si es el único
# es candidato a outlier
"""

data = {
    "salario": [30000, 32000, 31000, 1000000]
}

df = pd.DataFrame(data)

# Detectar outliers:
print(df.describe())

"""
df.describe() nos muestra información básica de los datos:

| estadístico | significado         |
| ----------- | ------------------- |
| count       | número de valores   |
| mean        | media               |
| std         | desviación típica   |
| min         | valor mínimo        |
| 25%         | primer cuartil (Q1) |
| 50%         | mediana (Q2)        |
| 75%         | tercer cuartil (Q3) |
| max         | valor máximo        |

Se puede analizar:

1- mirando percentiles:

    ej: edad:
    25% = 22
    50% = 30
    75% = 35
    max = 120
    
    sospechas outlier porque:
    75% de los datos están ≤ 35
    pero el máximo es 120
    
2- Comparar media vs mediana

    Si se da un caso similar a este:
        mean = 40
        50% (mediana) = 30
        max = 500
    
    La media está inflada,probable presencia de outliers grandes

3- Usando desviación estándar:

    si std es muy grande comparado con la media, es que los datos están muy dispersos,
    lo que puede indicar outliers

"""

# Forma 1: Regla del IQR

"""
Usando quartiles

Para ello calculamos IQR que es igual a IQR=Q3−Q1
    Un valor es outlier si:
    x<Q1−1.5⋅IQRox>Q3+1.5⋅IQR
    
El IQR (Interquartile Range) es el rango donde está el 50% central de los datos (75% - 25%).
Q1 y Q3 contienen el centro de los datos, por lo que el IQR mide la “anchura normal”.
Todo lo que se sale mucho es sospechoso

Ventajas
    -Muy robusto (no le afectan mucho los extremos)
    -Muy usado en ML real
    -Funciona bien con distribuciones no normales

Desventajas
    -No tiene en cuenta la forma global de la distribución

"""

Q1 = df["salario"].quantile(0.25)
Q3 = df["salario"].quantile(0.75)

IQR = Q3 - Q1

limite_inferior = Q1 - 1.5 * IQR
limite_superior = Q3 + 1.5 * IQR

outliers = df[
    (df["salario"] < limite_inferior) |
    (df["salario"] > limite_superior)
]

print("Outliers detectados:")
print(outliers)


# Forma 2: Z-Score (basado en desviaciones estándar)

"""
Mide cuántas desviaciones estándar se aleja un valor de la media.

La desviación típica => z= (x−μ)/σ

Si |z| > 3 suele ser outlier, porque:
    z = 0 → valor normal (en la media)
    z = 2 → algo alejado
    z = 3+ → muy raro
    
Ventajas:
Muy simple
Funciona bien con datos normales (distribución gaussiana)

Desventajas:
Muy sensible a outliers (la media y desviación se distorsionan)
No funciona bien si los datos no son normales

"""

from scipy import stats

df["z_score"] = np.abs(stats.zscore(df["salario"]))

outliers = df[df["z_score"] > 3]

print(outliers)

#Comparación directa

"""

comparar directamente con:

mínimos
máximos
percentiles
reglas del negocio

Ventajas
    Muy interpretable
    Usa lógica del problema
    Ideal cuando conoces el dominio
Desventajas
    No automático
    Depende del conocimiento humano

"""

# Ponemos reglas manuales como maximos y minimos salarios:
lower_limit = 20000
upper_limit = 80000

outliers = df[(df["salario"] < lower_limit) | (df["salario"] > upper_limit)]

print(outliers)


#Mostar datos visuales:

import matplotlib.pyplot as plt

plt.boxplot(df["salario"])
plt.show()

