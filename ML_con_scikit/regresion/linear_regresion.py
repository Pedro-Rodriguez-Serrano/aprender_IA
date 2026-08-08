"""
La regresión lineal se utiliza cuando queremos predecir un número.

Clasificación → predice una categoría: Sí/No, Spam/No spam
Regresión → predice un valor numérico: salario, precio, ventas, temperatura
"""

#Ej: Tratar de predecir cuanto gastará un cliente:
import pandas as pd

data = {
    "edad": [20, 25, 30, 35, 40, 45, 50],
    "salario": [20000, 25000, 30000, 35000, 40000, 50000, 60000],
    "gasto": [500, 700, 900, 1200, 1500, 1800, 2200]
}

df = pd.DataFrame(data)

print(df)

#Separar x e y
X = df[["edad", "salario"]]

y = df["gasto"]

#Crear train y test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

#Crear modelo:
from sklearn.linear_model import LinearRegression

modelo = LinearRegression()

#Entrenar
modelo.fit(X_train, y_train)

#Predecir:
predicciones = modelo.predict(X_test)

print("Predicción: ",predicciones, "para \n", X_test)

#Predecir un cliente nuevo

nuevo_cliente = pd.DataFrame({
    "edad": [38],
    "salario": [42000]
})

prediccion = modelo.predict(nuevo_cliente)

print("Predicción cliente \n", nuevo_cliente, " = ",prediccion)

#Evaluar la regresión

#Cuanto nos equivocamos de media
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(
    y_test,
    predicciones
)

print("MAE:", mae)

"""
i obtenemos:

MAE = 120

significa aproximadamente:

El modelo se equivoca unos 134 € de media.
"""

from sklearn.metrics import mean_squared_error

mse = mean_squared_error(
    y_test,
    predicciones
)

print("MSE:", mse)

"""
Eleva los errores al cuadrado, por lo que penaliza mucho los errores grandes.
"""

#R²
from sklearn.metrics import r2_score

r2 = r2_score(
    y_test,
    predicciones
)

print("R²:", r2)

"""
R² ≈ 1 → modelo explica muy bien los datos
R² ≈ 0 → explica poco
R² < 0 → puede ser peor que una predicción muy básica basada en la media
"""

"""
NOTA:
LinearRegression no necesita obligatoriamente escalado para funcionar correctamente.
"""