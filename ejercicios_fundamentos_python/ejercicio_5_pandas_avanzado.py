import pandas as pd

df = pd.DataFrame({
    "Departamento": ["IT", "IT", "HR", "HR", "Sales", "Sales"],
    "Empleado": ["Ana", "Luis", "Marta", "Pedro", "Juan", "Sara"],
    "Salario": [2000, 2500, 1800, 1700, 2200, 2100],
    "Años": [2, 5, 3, 1, 4, 2]
})

#Salario medio por departamento
salario_medio_departamento = df.groupby("Departamento")["Salario"].mean()
print("Salario medio agrupado por salario: \n", salario_medio_departamento)

#Empleados con mas de 3 años
print("Los empleados con mas de 3 años es: \n", df[df["Años"] > 3])

# Crear columna senior
df["Senior"] = df["Años"] >= 3
print("Columna senior añadida \n",df)

#Salario medio por senior y no senior
print("Salario medio por senior y no senior: \n",df.groupby("Senior")["Salario"].mean())

# Rankin empleados en base a salario:
print("Ranking empleados por salario: \n",df.sort_values("Salario", ascending=False))
