import pandas as pd

df = pd.DataFrame({
    "Nombre": ["Ana", "Luis", "Marta", "Juan"],
    "Nota": [8, 6, 4, 9]
})

# Crea una columna “Nota_Categoria”:

def categoria(nota):
    if nota >= 7:
        return "Alta"
    elif nota >= 5:
        return "Media"
    else:
        return "Baja"

df["Categoria"] = df["Nota"].apply(categoria) # Aply recibe una función y le pasa el parametro de cada nota
print("df tras añadir nota_categoria:\n",df)

# Calcular media de nota por categoria
nota_media = df.groupby("Categoria")["Nota"].mean()
print("La nota media es: \n",nota_media)

# Calcular nota máxima por categoria
nota_max = df.groupby("Categoria")["Nota"].max()
print("Nota máxima por categoria: \n",nota_max)

# Calcular cantidad de alumnos por categoria
num_alumnos = df.groupby("Categoria").count()
print("Numero alumnos por categoria: \n",num_alumnos)

# Crear tabla resumen por categoría
df.pivot_table(values="Nota", index="Categoria", aggfunc="mean")

# unir tablas
df1 = pd.DataFrame({
    "ID": [1, 2, 3],
    "Nombre": ["Ana", "Luis", "Marta"]
})

df2 = pd.DataFrame({
    "ID": [1, 2, 3],
    "Nota": [8, 6, 9]
})

df_merge = pd.merge(df1, df2, on="ID")
print("Los datos fusionados son: \n",df_merge)


df = pd.DataFrame({
    "Categoria": ["Alta", "Media", "Alta", "Baja"],
    "Nota": [8, 6, 9, 4]
})

#resumen de la tabla nota por categoria:
# Agrupa por "Categoria" (index), Calcula la media de Notas (values, aggfunc)
print("Resumen tabla nota por categoria:\n",df.pivot_table(values="Notas", index="Categoria", aggfunc="mean"))

#Usar valor nan (dato que falta)
import numpy as np

df = pd.DataFrame({
    "Nombre": ["Ana", "Luis"],
    "Nota": [8, np.nan]
})

df["Nota"].fillna(0)        # reemplazar
df["Nota"].mean()          # ignora nan automáticamente
df.dropna()                # eliminar filas con nan