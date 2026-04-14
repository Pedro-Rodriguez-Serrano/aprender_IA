import numpy as np

# Ejercicio 1:
# Crear un array con los números del 1 al 10.

arr = np.arange(1,10,dtype=int)
print(arr)

# Ejercicio 2:
arr * 2   # multiplica todos los elementos
arr + 5   # suma 5 a todos los elementos

# Ejercicio 3:
# estadistica básica: Media, mediana, desviación estándar

data = np.array([10, 20, 30, 40, 50])

print("Media:", np.mean(data))
print("Mediana:", np.median(data))
print("Desviación estándar:", np.std(data))

# Ejercicio 4:
# Max, Min, rango

maximo = np.max(data)
minimo = np.min(data)
rango = maximo - minimo

print("Maximo: ",maximo)
print("minimo: ", minimo)
print("Rango: ", rango)

# Ejercicio 5:
# calcular media de notas, aprobados y porcentaje aprobados

notas = np.array([7, 5, 9, 6, 8, 10, 4, 6])

media = np.mean(notas)
aprobados = np.sum(notas >= 5)
porcentaje = (aprobados / len(notas)) * 100

print("La media de las notas es: ",media)
print("Los aprobados son: ",aprobados)
print("El porcentaje de aprobados es: ",porcentaje)

# Ejercicio 6:
# Normalización de datos

data = np.array([50, 60, 70, 80, 90])

media = np.mean(data)
std = np.std(data)

normalizado = (data - media) / std

print("Datos normalizados: ",normalizado)

# Ejercicio 7:
# Correlación:

x = np.array([1,2,3,4,5])
y = np.array([2,4,6,8,10])

correlacion = np.corrcoef(x, y)

print("La correlación entre X: ", x, " Y: ", y, " es: ", correlacion)

# Ejercicio 8:
# sistema de ecuaciones

"""
Ecuación: 
2x + y = 5
x + 3y = 6
"""

A = np.array([[2, 1],
              [1, 3]])

b = np.array([5, 6])

sol = np.linalg.solve(A, b)
print("Solución ecuación 2x + y = 5 \n"
      "x + 3y = 6 -> ",sol)