import pandas as pd
import numpy as np

# Ejercicio 1: detectar y tratar valores faltantes

# Dataset:

data = {
    "edad": [25, 30, np.nan, 40, 35],
    "salario": [30000, 40000, 50000, np.nan, 45000],
    "ciudad": ["Madrid", "Barcelona", "Madrid", np.nan, "Valencia"]
}

df = pd.DataFrame(data)

# detectamos valores nulos:
print("Valores nulos: \n",df.isnull())
print("Nulos por columna: ",df.isnull().sum())

# Opción 1: borrar nulos
df_drop = df.dropna()
print("Dataset con nulos: \n",df)
print("Dataset sin nulos: \n",df_drop)

# Opción 2: modificarlos (preferible)

# Los datos numéricos los sustituimos por las medias del campo
df["edad"] = df["edad"].fillna(df["edad"].mean())
df["salario"] = df["salario"].fillna(df["salario"].mean())

# Otros datos los sustituimos por el mas usado
df["ciudad"] = df["ciudad"].fillna(df["ciudad"].mode()[0])

# De esta forma evitamos perder información al no eliminar demasiada información,
# y al usar medias y valores mas usados no distorsionamos mucho los datos.
# La idea es no introducir sesgos fuertes ni información falsa.

# Resultado limpio
print("Datos limpios por metodo 2: \n",df)

