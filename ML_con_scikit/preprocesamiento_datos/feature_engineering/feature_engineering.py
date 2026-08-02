"""
Feature Engineering consiste en tomar los datos originales y convertirlos en datos
 más útiles para el modelo.
"""
import pandas as pd

data = {
    "edad": [20, 30, 40],
    "salario": [30000,45000,80000]
}

df = pd.DataFrame(data)

## Ejemplo 1: Crear una nueva variable

df["salario_por_edad"] = df["salario"] / df["edad"]

print(df)

## Ejemplo 2: Extraer información de una fecha

df = pd.DataFrame({
    "fecha": ["2024-01-15", "2024-02-20", "2024-03-10"],
    "ventas": [100, 150, 200]
})

df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")

df["año"] = df["fecha"].dt.year
df["mes"] = df["fecha"].dt.month
df["dia_semana"] = df["fecha"].dt.dayofweek

print(df)

## Ejemplo 3: Agrupar categorías y añadir columna

import pandas as pd

df = pd.DataFrame({
    "profesion": ["Ingeniero", "Programador", "Desarrollador", "Abogado"]
})

agrupacion = {
    "Ingeniero": "Tecnología",
    "Programador": "Tecnología",
    "Desarrollador": "Tecnología"
}

df["categoria"] = df["profesion"].replace(agrupacion)

print(df)