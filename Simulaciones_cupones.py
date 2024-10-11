# Databricks notebook source
# MAGIC %md
# MAGIC ## Simulaciones Modelo Sellout

# COMMAND ----------

import mlflow
import matplotlib.pyplot as plt
import numpy as np
logged_model = 'runs:/72f0ee92e6e54e779904ac8b715c6f85/model' #Hay que cambiarlo cada ejecución de modelo 

# Load model as a PyFuncModel.
loaded_model = mlflow.sklearn.load_model(logged_model)

# COMMAND ----------

coef_cupones=loaded_model.coef_[15:26]

# COMMAND ----------

coef_cupones

# COMMAND ----------

np.eye(11)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Efecto medio de meter un euro más en cupon sobre sellout (lag de cupones)

# COMMAND ----------

np.dot(np.eye(11),coef_cupones).sum()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Simulación 0

# COMMAND ----------

# MAGIC %md
# MAGIC UTILIZAMOS los coeficientes y las varianzas de los parámetros obtenidos de un modelo de regresión bayesiana.

# COMMAND ----------

sigma_cupones= np.diag(loaded_model.sigma_) #Varianzas de los parámetros

# COMMAND ----------

sigma_cupones=sigma_cupones[15:26]


# COMMAND ----------

sigma_cupones

# COMMAND ----------

np.random.normal(coef_cupones,np.sqrt(sigma_cupones))

# COMMAND ----------

results=[]
for i in range (300):
    c1=np.random.normal(coef_cupones,np.sqrt(sigma_cupones))
    r=np.dot(np.eye(11),c1).sum()
    results.append(r)

# COMMAND ----------

results=np.array(results)

# COMMAND ----------

results

# COMMAND ----------

plt.figure(figsize=(10, 6))
plt.hist(results, bins=20, color='skyblue', edgecolor='black')

# Añadir etiquetas y título
plt.xlabel('Valores')
plt.ylabel('Frecuencia')
plt.title('Intervalo de confianza')

# Mostrar el gráfico
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Simulación 1

# COMMAND ----------

# MAGIC %md
# MAGIC Mejoras: integración de variables polinómicas de coupondiscountLag y la polinomica en sí 

# COMMAND ----------

import mlflow
import matplotlib.pyplot as plt
import numpy as np
logged_model = 'runs:/02ce20eac7d8435e993c36c7b7058004/model' #Hay que cambiarlo cada ejecución de modelo 

# Load model as a PyFuncModel.
loaded_model = mlflow.sklearn.load_model(logged_model)

# COMMAND ----------

# MAGIC %md
# MAGIC ####Media de sellout

# COMMAND ----------

coef_cupones_1=loaded_model.coef_[25:58]

# COMMAND ----------

coef_cupones_1

# COMMAND ----------

np.dot(np.eye(33),coef_cupones_1).sum()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Intervalo de confianza de aumentar un euro en cantidad de cupon

# COMMAND ----------

sigma_cupones_1= np.diag(loaded_model.sigma_) #Varianzas de los parámetros

# COMMAND ----------

sigma_cupones_1=sigma_cupones_1[25:58]

# COMMAND ----------

np.random.normal(coef_cupones_1,np.sqrt(sigma_cupones_1))

# COMMAND ----------

results_1=[]
for i in range(300):
    c2= np.random.normal(coef_cupones_1,np.sqrt(sigma_cupones_1))
    r=np.dot(np.eye(33),c2).sum()
    results_1.append(r)


# COMMAND ----------

sigma_cupones_1.shape

# COMMAND ----------

results_1=np.array(results_1)

# COMMAND ----------

results_1

# COMMAND ----------

plt.figure(figsize=(10, 6))
plt.hist(results_1, bins=20, color='salmon', edgecolor='black')

# Añadir etiquetas y título
plt.xlabel('Valores')
plt.ylabel('Frecuencia')
plt.title('Intervalo de confianza')

# Mostrar el gráfico
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Distribución de los coeficientes

# COMMAND ----------

plt.figure(figsize=(10, 6))
plt.bar(range(len(coef_cupones_1)), coef_cupones_1, color='skyblue', edgecolor='black')
plt.title('Coeficientes del Modelo')
plt.xlabel('Índice del Coeficiente')
plt.ylabel('Valor del Coeficiente')
plt.xticks(range(len(coef_cupones_1)))
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Simulación 2

# COMMAND ----------

# MAGIC %md
# MAGIC Mejoras: incremento manual de 10 euros en cantidad de cupon. Tenemos que tener en cuenta que queremos ver el efecto en lagueadas y en polinómicas. 

# COMMAND ----------

import numpy as np

def generate_data_for_simulation(num_lags, coupon_level, poly_grade):
    out_list = [np.eye(num_lags) * (coupon_level ** k)
                for k in range(1, poly_grade + 1)]
    return np.concatenate(out_list, axis=1)



# COMMAND ----------


data = [generate_data_for_simulation(11, cl, 3)
        for cl in range(1, 3, 1)]
for i, d in enumerate(data, start=1):
    print(f"Generar simulaciones para el nivel {i}:")
    print(d)
    print("\\n")


# COMMAND ----------

results = []

# Iterar sobre las matrices generadas
for i, matrix in enumerate(data, start=1):
    nivel = i
    print(f"Generar simulaciones para el nivel {nivel}:")
    print(matrix)
    print("\\n")
    
    # Calcular el producto punto y la suma para cada matriz generada
    r = np.dot(matrix, coef_cupones_1).sum()
    results.append((nivel, r))

# Imprimir los resultados
for nivel, resultado in results:
    print(f"Resultado para el nivel {nivel}: {resultado}")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Simulación 3 ( sin cubo) PRINCIPAL

# COMMAND ----------

import mlflow
import matplotlib.pyplot as plt
import numpy as np
logged_model_2 = 'runs:/333396801b184ff7b0d4c00ea9185f3d/model' #Hay que cambiarlo cada ejecución de modelo 

# Load model as a PyFuncModel.
loaded_model_2 = mlflow.sklearn.load_model(logged_model_2)

# COMMAND ----------

loaded_model_2.coef_

# COMMAND ----------

coef_cupones_2=loaded_model_2.coef_[17:39]

# COMMAND ----------

sigma_cupones_2= np.diag(loaded_model_2.sigma_)

# COMMAND ----------

sigma_cupones_2=sigma_cupones_2[17:39]

# COMMAND ----------

sigma_cupones_2.shape

# COMMAND ----------

coef_cupones_2.shape


# COMMAND ----------

plt.figure(figsize=(10, 6))
plt.bar(range(len(coef_cupones_2)), coef_cupones_2, color='skyblue', edgecolor='black')
plt.title('Coeficientes del Modelo')
plt.xlabel('Índice del Coeficiente')
plt.ylabel('Valor del Coeficiente')
plt.xticks(range(len(coef_cupones_2)))
plt.show()

# COMMAND ----------

import numpy as np

def generate_data_for_simulation(num_lags, coupon_level, poly_grade):
    out_list = [np.eye(num_lags) * (coupon_level ** k)
                for k in range(1, poly_grade + 1)]
    return np.concatenate(out_list, axis=1)

# COMMAND ----------

data = [generate_data_for_simulation(11, cl, 2)
        for cl in range(1, 12, 2)]
for i, d in enumerate(data, start=1):
    print(f"Generar simulaciones para el nivel {i}:")
    print(d)
    print("\\n")

# COMMAND ----------

results = []

# Iterar sobre las matrices generadas
for i, matrix in enumerate(data, start=1):
    euros = i
    
    # Calcular el producto punto y la suma para cada matriz generada
    r = np.dot(matrix, coef_cupones_2).sum()
    results.append((euros, r))

# Imprimir los resultados
for nivel, resultado in results:
    print(f"Resultado media : {resultado}")

# COMMAND ----------

results_por_nivel = []

# Iterar sobre las matrices generadas
for matrix in data:
    resultados_nivel = []
    for _ in range(300):
        c2 = np.random.normal(coef_cupones_2, np.sqrt(sigma_cupones_2))
        r = np.dot(matrix, c2).sum()
        resultados_nivel.append(r)
    results_por_nivel.append(resultados_nivel)

# Crear el boxplot
plt.figure(figsize=(10, 6))

# Etiquetas para el eje x según los incrementos en euros de descuento
labels = [f"{i} euro{'s' if i != 1 else ''}" for i in [1, 2, 4, 6, 8, 10]]
plt.boxplot(results_por_nivel, labels=labels)

plt.title('Simulaciones de incrementar en euros de descuento por nivel')
plt.xlabel('Niveles')
plt.ylabel('Simulaciones')
plt.grid(True)
plt.show()



# COMMAND ----------

# MAGIC %md
# MAGIC #### Simulación 4 ( PARA 2024)

# COMMAND ----------

import mlflow
import matplotlib.pyplot as plt
import numpy as np
logged_model_3 = 'runs:/1805b4ca14404868966e5f9d65375724/model' #Hay que cambiarlo cada ejecución de modelo 

# Load model as a PyFuncModel.
loaded_model_3 = mlflow.sklearn.load_model(logged_model_3)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Distribución simulación cupon % vs fijo

# COMMAND ----------

a=loaded_model_3.coef_[16]
a

# COMMAND ----------

sigma_cupones_porcentual= np.diag(loaded_model_3.sigma_)

# COMMAND ----------

sigma_cupones_porcentual=sigma_cupones_porcentual[16]

# COMMAND ----------

np.random.normal(a,np.sqrt(sigma_cupones_porcentual))

# COMMAND ----------

results_porcentual=[]
for i in range (300):
    c=np.random.normal(a,np.sqrt(sigma_cupones_porcentual))
    r=np.dot(np.eye(1),c).sum()
    results_porcentual.append(r)

# COMMAND ----------

plt.figure(figsize=(10, 6))
plt.hist(results_porcentual, bins=20, color='skyblue', edgecolor='black')

# Añadir etiquetas y título
plt.xlabel('Valores')
plt.ylabel('Frecuencia')
plt.title('Intervalo de confianza de coeficiente "cupon_porcentual"')

# Mostrar el gráfico
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Distribución variables de estacionalidad

# COMMAND ----------

coef_season_otoño=loaded_model_2.coef_[0]
coef_season_primav=loaded_model_2.coef_[1]
coef_season_verano=loaded_model_2.coef_[2]

# COMMAND ----------

coef_season_otoño

# COMMAND ----------

sigma_season= np.diag(loaded_model_2.sigma_)

# COMMAND ----------

sigma_cupones_otoño=sigma_season[0]
sigma_cupones_primav=sigma_season[1]
sigma_cupones_verano=sigma_season[2]

# COMMAND ----------

n_simulaciones = 300
resultados_otoño = []
resultados_primav = []
resultados_verano = []

# Realizar las simulaciones
for _ in range(n_simulaciones):
    c_otoño = np.random.normal(coef_season_otoño, np.sqrt(sigma_cupones_otoño))
    c_primav = np.random.normal(coef_season_primav, np.sqrt(sigma_cupones_primav))
    c_verano = np.random.normal(coef_season_verano, np.sqrt(sigma_cupones_verano))
    
   
    r_otoño = np.dot(np.eye(1), c_otoño).sum()
    r_primav = np.dot(np.eye(1), c_primav).sum()
    r_verano = np.dot(np.eye(1), c_verano).sum()
    
    
    resultados_otoño.append(r_otoño)
    resultados_primav.append(r_primav)
    resultados_verano.append(r_verano)

data = [resultados_otoño, resultados_primav, resultados_verano]
plt.boxplot(data, labels=['Otoño', 'Primavera', 'Verano'])
plt.title('Distribuciones de los coeficientes por estación')
plt.ylabel('Valor del coeficiente')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Simulaciones Modelo Referencias
# MAGIC
# MAGIC

# COMMAND ----------

import mlflow
import matplotlib.pyplot as plt
import numpy as np
logged_model_ref = 'runs:/a782e26dea69422b9e05b63fe6069439/model' #Hay que cambiarlo cada ejecución de modelo 

# Load model as a PyFuncModel.
loaded_model_ref = mlflow.sklearn.load_model(logged_model_ref)

# COMMAND ----------

coef_refes=loaded_model_ref.coef_[17:39]

# COMMAND ----------

sigma_refes=np.diag(loaded_model_ref.sigma_)

# COMMAND ----------

sigma_refes=sigma_refes[17:39]

# COMMAND ----------

np.random.normal(coef_refes,np.sqrt(sigma_refes))

# COMMAND ----------

results_refes=[]
for i in range (300):
    c=np.random.normal(coef_refes,np.sqrt(sigma_refes))
    r=np.dot(np.eye(22),c).sum()
    results_refes.append(r)

# COMMAND ----------

plt.figure(figsize=(10, 6))
plt.hist(results_refes, bins=20, color='pink', edgecolor='black')

# Añadir etiquetas y título
plt.xlabel('Valores')
plt.ylabel('Frecuencia')
plt.title('Intervalo de confianza de coeficiente "cupon_porcentual"')

# Mostrar el gráfico
plt.show()

# COMMAND ----------

data = [generate_data_for_simulation(11, cl, 2)
        for cl in range(1, 12, 2)]
for i, d in enumerate(data, start=1):
    print(f"Generar simulaciones para el nivel {i}:")
    print(d)
    print("\\n")

# COMMAND ----------

results_por_refe = []

# Iterar sobre las matrices generadas
for matrix in data:
    resultados_por_refe = []
    for _ in range(300):
        c2 = np.random.normal(coef_refes, np.sqrt(sigma_refes))
        r = np.dot(matrix, c2).sum()
        resultados_por_refe.append(r)
    results_por_refe.append(resultados_por_refe)

# Crear el boxplot
plt.figure(figsize=(10, 6))

# Etiquetas para el eje x según los incrementos en euros de descuento
labels = [f"{i} euro{'s' if i != 1 else ''}" for i in [1, 2, 4,6,8,10]]
plt.boxplot(results_por_refe, labels=labels)

plt.title('Simulaciones de incrementar en euros de descuento por nivel')
plt.xlabel('Niveles')
plt.ylabel('Simulaciones')
plt.grid(True)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC