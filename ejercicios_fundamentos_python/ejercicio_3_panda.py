import pandas as pd
from color_consola import ColorConsola as cc

df = pd.DataFrame({
    "Nombre": ["Ana", "Luis", "Marta"],
    "Edad": [20, 22, 19],
    "Nota": [8, 6, 9]
})

print(df)

# Mostrar solo la columna de los nombres de los que tengan mas o igual de 7 de nota:
print("Nombres de alumnos con 7 o mas: \n",df.loc[df["Nota"] >= 7, "Nombre"])

# Obtener un solo valor concreto (Fila 0, columna 2)
print("fila 0 columna 2: ",df.iloc[0, 2])

# Obtener nombre de Ana
print("La Nota de ana es: ",df.loc[df["Nombre"] == "Ana", "Nota"])

# Obtener datos de ana limpios:
# Mejor opcion iloc 👇
print("La Nota de ana en formato limpio usando iloc es: ",
      df.loc[df["Nombre"] == "Ana", "Nota"].iloc[0])

print("La Nota de ana en formato limpio usando values es: ",
      df.loc[df["Nombre"] == "Ana", "Nota"].values[0])

# Obtener edad y nota de Ana
print("Edad y nota de Ana:\n ",df.loc[df["Nombre"] == "Ana", ["Edad", "Nota"]])

# Obtener edad y nota de Ana valores puros
fila = df.loc[df["Nombre"] == "Ana"]

edad = fila["Edad"].iloc[0]
nota = fila["Nota"].iloc[0]

print(cc.AZUL,f"La edad de Ana limpio es {edad} y la nota es {nota}")

# Obtener cabecera:
print(cc.CYAN,"Cabecera: ",df.head(1))

# Información dataframe:
print(cc.AMARILLO,"Información dataframe: ", df.info())

# Obtener estadisticas
print(cc.ROJO,"Estadisticas: ",df.describe())

# Crear nuevo campo:
df["DobleNota"] = df["Nota"] * 2
print(cc.SUBRAYADO,cc.BLANCO,df)

# Medias de nota por edad:
print(cc.RESET,cc.CYAN,
      "Media notas por edad: \n",df.groupby("Edad")["Nota"].mean())

# reiniciar df
df = pd.DataFrame({
    "Nombre": ["Ana", "Luis", "Marta"],
    "Edad": [20, 22, 19],
    "Nota": [8, 6, 9]
})

# Añadir filas
df.loc[len(df)] = ["Carlos", 21, 7]
df.loc[len(df)] = ["Carlos", 21, 7]

print(cc.RESET,"Antes de borrar duplicados:\n", df)

# Eliminar duplicados
df.drop_duplicates(inplace=True)

print("Después de borrar duplicados:\n", df)

# Guardar en csv
df.to_csv("datos.csv", index=False)

# Leer csv
df_csv = pd.read_csv("datos.csv")
print("df desde csv \n",df_csv)

# Mostrar datos sin indice:
print("Datos con index=False \n",
      df.loc[df["Nombre"] == "Ana", ["Edad", "Nota"]].to_string(index=False))

# Suma de las notas por grupos de edades usando groupby
df.loc[len(df)] = ["Carlos1", 28, 8]
df.loc[len(df)] = ["Carlos2", 28, 7]
df.loc[len(df)] = ["Carlos3", 28, 9]
#Media de 28 debería ser 8 y suma 24
print("Suma de las notas agrupados por edad: \n", df.groupby("Edad")["Nota"].sum())
print("Media de las notas agrupados por edad: \n", df.groupby("Edad")["Nota"].median())