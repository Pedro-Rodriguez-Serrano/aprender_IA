import pandas as pd
import numpy as np

data = {
    "ciudad": ["madrid", "Madrid", "MADRID", "Barcelona"]
}

df = pd.DataFrame(data)

# Normalizar texto (ajustar datos a minúsculas)
df["ciudad"] = df["ciudad"].str.lower()
print("Datos normalizados: \n",df)