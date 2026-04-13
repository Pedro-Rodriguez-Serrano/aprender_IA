# El objetivo de este ejercicio es generar un modelo que
# detecte ciertas palabras como spam. Se le enseña
# que palabras como "ganar dinero rapido" = spam

import pandas as pd

#dataset de frases que son y no son spam
data = {
    "mensaje": [
        "gana dinero rapido",
        "oferta limitada compra ahora",
        "hola como estas",
        "nos vemos mañana",
        "trabaja desde casa gana dinero",
        "reunion de equipo hoy"
    ],
    "spam": [1, 1, 0, 0, 1, 0]
}

# En panda convierte datos (diccionario) en un formato estilo Excel para procesarlo
data_frame = pd.DataFrame(data)
print(data_frame)

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data_frame["mensaje"])
#print(f"Los datos vectorizados de data_frame mensaje son: {X}")
#print("Los datos sin vectorizar de data_frame mensaje son: ",data_frame["mensaje"])
y = data_frame["spam"]

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

modelo = MultinomialNB()
modelo.fit(X_train, y_train)

accuracy = modelo.score(X_test, y_test)
print("Precisión:", accuracy)

nuevo = ["gana dinero desde casa"]
nuevo_vector = vectorizer.transform(nuevo)

prediccion = modelo.predict(nuevo_vector)

print("¿Es spam?", prediccion[0])