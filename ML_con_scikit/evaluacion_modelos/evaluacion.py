"""
Conceptos que sirven para evaluar modelo:

Matriz de confusión: tabla que usamos para evaluar qué tan bien funciona un
                     modelo de clasificación.
                    La matriz compara:
                    - lo que realmente ocurrió (y_test)
                    - lo que el modelo predijo (y_pred)
                    La matriz sirve para mostrar los verdaderos positivos que acertó,
                    los verdaderos negativos y también los falsos negativos o positivos,
                    por lo que sirve para evaluar el modelo e identificar los errores que
                    comete.

Accuracy: porcentaje total de predicciones correctas.
          Ejemplo: Accuracy = 90 / 100 = 0.90 → 90 % de accuracy.
          El problema de acuracy es que si por ejemplo tenemos un caso en el que el 90%
          es positivo, si el modelo marca positivo, el acuracy será del 90%, pero en realidad
          no ha acertado ni un solo negativo, por lo que el modelo no funciona.

Precision: Evalua: de todas las veces que el modelo dijo "positivo", ¿cuántas eran realmente
           positivas?
           Por ejemplo:

           El modelo predijo:

                20 clientes → Compra

            De esos:

                15 → realmente compraron
                5 → no compraron

            Entonces:

                TP = 15
                FP = 5
                Precision = 15 / (15 + 5)
                        = 0.75

                → 75 %

            Interpretación: cuando el modelo dice "compra", tiene razón el 75 % de las veces.

           Este método_es especialmente importante cuando los falsos positivos son costosos.

            Ejemplo: detector de spam.

            No quieres que un email importante sea marcado incorrectamente como spam.

Recall: De todos los positivos reales, ¿cuántos consiguió detectar el modelo?

        Ejemplo:

            Existen:

                100 personas enfermas

            El modelo detecta:

                90 enfermas

            Pero se le escapan:

                10 enfermas

            Entonces:

                TP = 90
                FN = 10
                Recall = 90 / (90 + 10)
                    = 0.90

             → 90 %

            Interpretación

                El modelo encuentra el 90 % de los casos positivos reales.

        Recall es muy importante cuando los falsos negativos son costosos.

            Ejemplo: detección de cáncer.

                Es preferible revisar a alguien sano por error:

                    "Falso positivo"

                que no detectar a una persona enferma:

F1-Score: combina Precision y Recall.
          Es útil cuando:

            -Las clases están desbalanceadas.
            -Te importan tanto Precision como Recall.

         Ejemplo

            Supongamos:

                Precision = 0.90
                Recall = 0.40

            Aunque la Precision es muy buena, el Recall es bastante bajo.

            El F1-Score será también limitado, porque el modelo no está funcionando bien en ambos aspectos.

            Para tener un F1 alto, necesitas un buen equilibrio entre:

                Precision alta
                       +
                Recall alto
                       ↓
                 F1-Score alto

classification_report: Sirve para obtener varias métricas en una sola línea. Incluye las anteriores pero
                        también otros si quieres hilar más fino (Evalua por clase, cuantos ejemplos por
                        clase había (Support), Calcula la media de las métricas de cada clase dando la misma
                        importancia a todas las clases (Macro Average), media, pero teniendo en cuenta cuántos
                         ejemplos hay de cada clase (Weighted Average)).

Macro vs Weighted:

|                                   | Macro Avg                          | Weighted Avg            |
| --------------------------------- | ---------------------------------- | ----------------------- |
| Tiene en cuenta cantidad de datos | ❌                                 | ✅                      |
| Todas las clases pesan igual      | ✅                                 | ❌                      |
| Útil para clases desbalanceadas   | Sí, para analizar clases por igual | Sí, para resumen global |


"""

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# 1. Crear dataset

data = {
    "edad": [
        20, 21, 22, 23, 24, 25,
        35, 36, 38, 40, 42, 45
    ],
    "salario": [
        20000, 21000, 22000, 23000, 24000, 25000,
        55000, 58000, 62000, 65000, 70000, 80000
    ],
    "compra": [
        False, False, False, False, False, False,
        True, True, True, True, True, True
    ]
}

df = pd.DataFrame(data)

# 2. Separar características y objetivo

X = df[["edad", "salario"]]
y = df["compra"]

# 3. Separar entrenamiento y prueba

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# 4. Crear y entrenar

modelo = KNeighborsClassifier(
    n_neighbors=3
)

modelo.fit(
    X_train,
    y_train
)

# 5. Predicciones

predicciones = modelo.predict(
    X_test
)

# 6. Accuracy

accuracy = accuracy_score(
    y_test,
    predicciones
)
print("Acuracy= ", accuracy)

# 7. Precision

precision = precision_score(
    y_test,
    predicciones
)

print("Precision: ",precision)

# 8. Recall

recall = recall_score(
    y_test,
    predicciones
)

print("Recall: ", recall)

# 9. F1 Score

f1 = f1_score(
    y_test,
    predicciones
)

print("F1 Score:",f1)

# 10. Matriz de confusión

matriz = confusion_matrix(
    y_test,
    predicciones
)

print("Matriz de confusión:", matriz)

# 11. Classification Report

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predicciones
    )
)

