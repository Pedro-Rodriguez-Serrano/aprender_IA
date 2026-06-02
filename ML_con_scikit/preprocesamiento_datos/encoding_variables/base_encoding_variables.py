
"""
Los modelos trabajan con números, no con texto, por eso hay que pasar palabras a números.

Ej:

ciudad
------
Madrid -> 0
Barcelona -> 1
Valencia -> 2

Para hacerlo en python usaremos LabelEncoder de sklearn

"""

import pandas as pd

data = {
    "ciudad": ["Madrid","Barcelona","Valencia"]
}

df = pd.DataFrame(data)

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df["ciudad"] = le.fit_transform(df["ciudad"])

print("DataFrame con LabelEncoder: \n",
      df["ciudad"])

"""
El problema:

El modelo puede interpretar:

    Valencia (2) > Madrid (1) > Barcelona (0)

como si hubiera una relación de magnitud.

Pero las ciudades no tienen orden.

Por eso normalmente no se usa para variables nominales.

Aquí se usa One-Hot Encoding.

Se suele usar con pandas.get_dummies()

"""

data = {
    "ciudad": ["Madrid","Barcelona","Valencia"]
}

df = pd.DataFrame(data)

df_encoded = pd.get_dummies(df, columns=["ciudad"])

print("DataFrame encoded con get_dummies:\n",df_encoded)

"""

Creará:
   edad  ciudad_Barcelona  ciudad_Madrid  ciudad_Valencia
0    25             False           True            False
1    30              True          False            False
2    35             False          False             True

Lo que significa que Madrid se convitió en: 
Barcelona=false
Madrid=true
Valencia=false

Pero en proyectos reales se suele usar otra opción mas limpia y eficiente:

OneHotEncoder()

generará :
[[0. 1. 0.]
 [1. 0. 0.]
 [0. 0. 1.]]

"""

from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder()

data = {
    "ciudad": ["Madrid", "Barcelona", "Valencia"]
}

df = pd.DataFrame(data)

from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse_output=False)

encoded = encoder.fit_transform(df[["ciudad"]])

print("DF con OneHotEncoder:\n",encoded)

# Mostrar nombres columnas:
print(encoder.get_feature_names_out())

"""
Crearemos un DataFrame final
"""

encoded_df = pd.DataFrame(
    encoded,
    columns=encoder.get_feature_names_out()
)

df_final = pd.concat(
    [df.drop("ciudad", axis=1), encoded_df],
    axis=1
)

print("DF final de One_Hot + nombres columnas para hacerlo mas visual):\n",df_final)

"""
Cuando usar cada uno?

Si hay muchas categorías One-Hot no es buena idea, requiere muchas columnas.

Para variables ordinales (Bajo, Medio, Alto) es mejor LabelEncoding

"""