import pandas as pd
import numpy as np

data = {
    "nombre": ["Ana", "Luis", "Ana", "Carlos"],
    "edad": [25, 30, 25, 40]
}

df = pd.DataFrame(data)

# Detectar duplicados

print("Los valores están duplicados?: \n",df.duplicated())

# Borrar duplicados:
df = df.drop_duplicates()
print("Dataset sin duplicados: \n",df)