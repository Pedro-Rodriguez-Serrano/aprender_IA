import numpy as np

"""
Base machine learning es algebra lineal (Vectores y Matrices)

Vectores: lista de números (columnas, filas...)

Matrices: tabla de números
"""

#Vectores:
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

print("v1: ",v1)
print("v2: ",v2)

"""
Las operaciones mas importantes son:

-Suma
-Producto escalar (multiplicación vectores)
-Producto de matrices
-Transpuesta (intercambiar filas por columnas en matriz)

"""

# suma vectores
print("suma v1 y v2: ",v1 + v2)

# producto escalar
print("Producto escalar v1*v2: ",np.dot(v1, v2))

# matrices
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print("Matriz A: \n",A)
print("Matriz B: \n",B)

# multiplicación de matrices
print("Multiplicación matriz A*B: \n",np.matmul(A, B))

"""
Por qué importa?:
Redes neuronales = multiplicaciones de matrices
Datos = matrices (filas = ejemplos, columnas = variables)
"""

