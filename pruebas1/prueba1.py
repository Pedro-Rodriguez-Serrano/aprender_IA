# Primero hay que instalar scikit-learn pandas con el comando en powersell:
# pip install scikit-learn pandas

"""
Este programa tratará de predecir si alguien aprueba o no en base a sus horas de estudio. (>3 = aprueba)
"""

#Creamos datos:

# Horas de estudio
X = [[1], [2], [3], [4], [5]]

# Resultado: 0 = suspende, 1 = aprueba
y = [0, 0, 0, 1, 1]

# importamos libreria para crear modelo y entrenarlo
from sklearn.linear_model import LogisticRegression

# Crea modelo
modelo = LogisticRegression()
#entrenamiento
modelo.fit(X, y)

# Mostrar lo que aprendió
m = modelo.coef_
b = modelo.intercept_
print("La Pendiente es: ",m)      # pendiente (m)
print("La b es: ",b) # b
print(f"La ecuación es y={m}*x + {b}")

# ¿Qué pasa si alguien estudia 3.5 horas?
prediccion = modelo.predict([[3.5]])

print(prediccion)

# Probar con mas datos:
print(modelo.predict([[1]]))  # debería ser 0
print(modelo.predict([[5]]))  # debería ser 1