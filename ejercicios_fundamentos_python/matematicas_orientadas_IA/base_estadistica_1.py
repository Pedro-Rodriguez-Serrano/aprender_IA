"""
Estadistica:

    Media (promedio)

    Varianza (mide qué tanto se dispersan los datos respecto a la media.)

    Desviación estándar (Es la raíz cuadrada de la varianza -> La varianza
        está en unidades “al cuadrado”, la desviación estándar vuelve a unidades normales)

    Distribuciones de datos: Describe cómo se repiten los valores.

     Tipos comunes:
        -Normal (campana de Gauss)
        -Uniforme (todos igual de probables)
        -Sesgada (hacia un lado)
        -Ejemplo: Alturas humanas: mayoría cerca de 1.70m y pocos muy bajos o muy altos
            → distribución normal

    Correlación : La relación entre dos variables (1 relación perfecta, 0 no relación,
    -1 negativa)
    Ejemplo:
        horas de estudio ↑ → notas ↑ → correlación positiva
        temperatura ↑ → consumo de calefacción ↓ → negativa
"""

import numpy as np

datos = np.array([10, 12, 23, 23, 16, 23, 21, 16])

print("Media:", np.mean(datos))
print("Varianza:", np.var(datos))
print("Desviación estándar:", np.std(datos))

#Correlación:

x = np.array([1,2,3,4])
y = np.array([2,4,6,8])

print("X:",x)
print("Y:",y)

corr = np.corrcoef(x, y)[0,1]
print("Correlación entre x e y: ",corr)