"""
La probabilidad es un número entre 0 y 1:

0 → imposible
1 → seguro
0.5 → 50% de probabilidad
Fórmula básica:

P(A) = casos favorables / casos posibles

Ejemplo simple: Lanzar una moneda:

Probabilidad de un suceso (Que salga A) =
casos favorables (1) / casos posibles ({cara, cruz} 2) = P(cara) = 1/2 = 0.5


Tipos básicos de probabilidad:

1. clásica: Todos los resultados tienen la misma probabilidad
Ej: dados, monedas

2. Probabilidad empírica
Basada en experimentos reales
Ej: “de 100 lanzamientos, salió cara 52 veces → 0.52”

3. Probabilidad condicional
Probabilidad de algo si ya ocurrió otra cosa

Ej: P(A∣B)

Ejemplo: probabilidad de llover si está nublado.

"""

# Ejemplo: simular una moneda

import random

n = 1000
caras = 0

for _ in range(n):
    if random.choice([0,1]) == 1:
        caras += 1

print(caras / n)

# con numpy:

import numpy as np

# 1000 lanzamientos, resultados 0 y 1
lanzamientos = np.random.choice([0,1], size=1000)
prob = np.mean(lanzamientos)

print("De los 1000 lanzamientos la media fue: ",prob)

