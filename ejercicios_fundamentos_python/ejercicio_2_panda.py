# ¿Qué hace Pandas?
    # Pandas = librería de Python para trabajar con datos en forma de tablas,
    # como si fueran Excel dentro de Python.
#Ejercicios pandas

#ej 1:
#Crea un DataFrame con esta información:
#Nombre: Ana, Luis, Marta
#Edad: 20, 22, 19
#Nota: 8, 6, 9

import pandas as pd

df = pd.DataFrame({
    "Nombre": ["Ana", "Luis", "Marta"],
    "Edad": [20, 22, 19],
    "Nota": [8, 6, 9]
})

print(df)

#Ej 2 mostrar columna edad

print("La columna edad contiene lo siguiente: \n"
,df["Edad"])

#Ej 3 Muestra solo los alumnos con nota mayor o igual a 7.
print("Los alumnos con nota mayor o igual a 7 son: \n",
df[df["Nota"] >= 7])

#ej 4 Crea una columna llamada "Aprobado" (True si nota >= 5, False si no).
df["Aprobado"] = df["Nota"] >= 5
print("Los aprobados son: ", df["Aprobado"])

# ej 5 modificar valores +1 todas las notas
df["Nota"] = df["Nota"] + 1
print("tras actualizar notas +1 el resultado es:\n"
,df)

#ej 6
#Calcular:
#media de notas
#nota máxima
#nota mínima
print("La media de las notas es:",df["Nota"].mean())
print("La nota máxima es: ",df["Nota"].max())
print("La nota mínima es: ", df["Nota"].min())

#ej 7 ordenar el dataframe por nota de mayor a menor
df_ordenado_amenor = df.sort_values(by="Nota", ascending=False)
print("las notas ordenadas de mayor a menor es: \n"
,df_ordenado_amenor)

df_ordenado_amayor = df.sort_values(by="Nota", ascending=True)
print("las notas ordenadas de menor a mauor es: \n"
,df_ordenado_amayor)

#ej 8 contar valores. Contar cuántos alumnos están aprobados
print("Los alumnos aprobados son: ",df["Aprobado"].value_counts())

print(df["Aprobado"].value_counts())

# ej 9 lectura tipo css
# Simular un archivo CSV y cárgalo.
from io import StringIO

data = """Nombre,Edad,Nota
Ana,20,8
Luis,22,6
Marta,19,9"""

df = pd.read_csv(StringIO(data))
print("datos a csv:",df)

# ej 10
#crea una columna "Nota_Ajustada" = Nota * 1.1
# filtrar los alumnos con nota ajustada >= 8
# calcula la media de edad de esos alumnos

df["Nota_Ajustada"] = df["Nota"] * 1.1

filtrado = df[df["Nota_Ajustada"] >= 8]

print(filtrado)
print(filtrado["Edad"].mean())
