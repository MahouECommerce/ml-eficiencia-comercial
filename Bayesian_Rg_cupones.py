# Databricks notebook source
# MAGIC ### Modelo Target SELLOUT

# MAGIC En este notebook desarrolaremos un modelo de regresión (Bayessian Ridge)
import sys
ruta_principal = '/Workspace/Users/clara.trujillosantosolmo@emeal.nttdata.com/.ide/ml-recommender-data-gen-14982321'
sys.path.insert(0, ruta_principal)

import pandas as pd
from sklearn.linear_model import BayesianRidge
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.io_utils import save_file_to_format, read_file_in_format
input_format="parquet"

import mlflow.pyfunc
# spark = SparkSession.builder.appName("SparkUDFExample").getOrCreate()
# MAGIC ####Paths

order_detail_sorted_path="/dbfs/mnt/MSM/cleaned_data/order_detail_model.parquet"
order_detail_sorted=read_file_in_format(order_detail_sorted_path, format=input_format)

df=order_detail_sorted.copy()

# MAGIC ####Label encoding de "tipologia"

df=pd.get_dummies(df, columns=['tipologia'], prefix='Tip')
df=pd.get_dummies(df, columns=['Origin'], prefix='Orig')
df=pd.get_dummies(df, columns=['Coupon_type'], prefix='Type')
df=pd.get_dummies(df, columns=['season'], prefix='season')

# MAGIC ### Filtro Sellout < 3000
df['Sellout'].describe()

df=df[df['Sellout']<= 2500]

df=df[df['year']==2023]
prueba=features[features.columns[17:39]]
prueba
coef_with_weight

# MAGIC %md
# MAGIC #### Comparación sellout inversión
df['Sellout'].sum()
np.dot(prueba,coef_no_weight[17:39]).sum()
df['CouponDiscountAmt'].sum()

# MAGIC ####Selección de columnas para el modelo



feat_cols = [
'season_Otoño',
'season_Primavera',
'season_Verano',
'DayCount',
'DayCount^2',
'Tip_Bar Tradicional',
'Tip_Cervecería',
'Tip_Discoteca',
'Tip_Establecimiento de Tapeo',
'Tip_No Segmentado',
'Tip_Noche Temprana',
'Tip_Pastelería/Cafetería/Panadería',
'Tip_Restaurante',
'Tip_Restaurante de Imagen',
'Tip_Restaurante de Imagen con Tapeo',
'Orig_OFFLINE',
'Type_Porcentual',
'CouponDiscountAmt',
'CouponDiscountAmt_LAG_1',
'CouponDiscountAmt_LAG_2',
'CouponDiscountAmt_LAG_3',
'CouponDiscountAmt_LAG_4',
'CouponDiscountAmt_LAG_5',
'CouponDiscountAmt_LAG_6',
'CouponDiscountAmt_LAG_7',
'CouponDiscountAmt_LAG_8',
'CouponDiscountAmt_LAG_9',
'CouponDiscountAmt_LAG_10',
'CouponDiscountAmt^2',
'CouponDiscountAmt_LAG_1_POLY_2',
'CouponDiscountAmt_LAG_2_POLY_2',
'CouponDiscountAmt_LAG_3_POLY_2',
'CouponDiscountAmt_LAG_4_POLY_2',
'CouponDiscountAmt_LAG_5_POLY_2',
'CouponDiscountAmt_LAG_6_POLY_2',
'CouponDiscountAmt_LAG_7_POLY_2',
'CouponDiscountAmt_LAG_8_POLY_2',
'CouponDiscountAmt_LAG_9_POLY_2',
'CouponDiscountAmt_LAG_10_POLY_2',
'Sellout_LAG_1',
'Sellout_LAG_2',
'Sellout_LAG_3',
'Sellout_LAG_4',
'Sellout_LAG_5',
'Sellout_LAG_6',
'Sellout_LAG_7',
'Sellout_LAG_8',
'Sellout_LAG_9',
'Sellout_LAG_10',
'Sellout_LAG_1_POLY_2',
'Sellout_LAG_2_POLY_2', 
'Sellout_LAG_3_POLY_2', 
'Sellout_LAG_4_POLY_2',
'Sellout_LAG_5_POLY_2', 
'Sellout_LAG_6_POLY_2', 
'Sellout_LAG_7_POLY_2',
'Sellout_LAG_8_POLY_2', 
'Sellout_LAG_9_POLY_2', 
'Sellout_LAG_10_POLY_2',
]
target_col = 'Sellout'
features = df[feat_cols]
target = df[target_col]




def imprimir_nombres_caracteristicas(feat_cols):
    for indice, caracteristica in enumerate(feat_cols):
        print(f"Índice {indice}: {caracteristica}")

# Llamada a la función con la lista feat_cols
imprimir_nombres_caracteristicas(feat_cols)



df['DayCount']=df['DayCount'].fillna(0)



na_counts = features.isna().sum()
na_columns = na_counts[na_counts > 0]
na_columns





# MAGIC ### Ajuste Sample_weight para desbalanceo de muestras



pedidos_con_cupon = order_detail_sorted[order_detail_sorted['CouponDescription'] != 'NoCupon']['Code'].value_counts()
pedidos_sin_cupon = order_detail_sorted[order_detail_sorted['CouponDescription'] == 'NoCupon']['Code'].value_counts()

# Calcular número de pedidos con y sin cupon
n_with_coupon = len(pedidos_con_cupon)
n_no_coupon = len(pedidos_sin_cupon)
weight_no_coupon = 1
weight_with_coupon = n_no_coupon / n_with_coupon



#Creación del array de pesos
feature_balance=features['CouponDiscountAmt']
sample_weight = np.where(feature_balance == 0, weight_no_coupon, weight_with_coupon)




# MAGIC ####Aplicación del modelo y entrenamiento



#FIT MODELO SIN PESOS
model_no_weight = BayesianRidge(fit_intercept=False)
model_no_weight.fit(features, target )
coef_no_weight = model_no_weight.coef_





# FIT modelo CON PESOS
model_with_weight = BayesianRidge(fit_intercept=False)
model_with_weight.fit(features, target, sample_weight=sample_weight)
coef_with_weight = model_with_weight.coef_





# # Comparar los coeficientes
# print("Coeficientes sin pesos:")
# print(coef_no_weight)
# print("\nCoeficientes con pesos:")
# print(coef_with_weight)

# # Calcular la diferencia en coeficientes
# coef_difference = coef_with_weight - coef_no_weight
# print("\nDiferencia en coeficientes:")
# print(coef_difference)




# MAGIC #### Predicciones



import mlflow
logged_model = 'runs:/37e401184e5f493ab102c85fad08b339/model' #Hay que cambiarlo cada ejecución de modelo 

# Load model as a df.
loaded_model = mlflow.sklearn.load_model(logged_model)
loaded_model_pyfunc = mlflow.pyfunc.load_model(logged_model)





# Predict on a Pandas DataFrame.
import pandas as pd
predictions= loaded_model_pyfunc.predict(pd.DataFrame(df))



y_real=df['Sellout'].values




# MAGIC ### Plot modelo 0




plt.figure(figsize=(10, 6))

plt.scatter(y_real, predictions, color='skyblue', label='Datos')
plt.plot([min(y_real), max(y_real)], [min(y_real), max(y_real)], color='red', linestyle='--', label='Ideal')

plt.xlabel('Valores Reales')
plt.ylabel('Valores Predichos')
plt.title('Predicciones vs Reales')
plt.legend(loc='upper left')

plt.show()




# MAGIC #### Distribución de los coeficientes de la regresión 



coef_with_weight



# with mlflow.start_run() as run:
#     # Crear el gráfico de dispersión
#     plt.figure(figsize=(10, 6))
#     plt.scatter(y_real, predictions, color='blue', label='Datos')
#     plt.plot([min(y_real), max(y_real)], [min(y_real), max(y_real)], color='red', linestyle='--', label='Ideal')

#     plt.xlabel('Valores Reales')
#     plt.ylabel('Valores Predichos')
#     plt.title('Predicciones vs Reales')
#     plt.legend(loc='upper left')

#     # Guarda el gráfico como un archivo temporal
#     plt.savefig('/tmp/predicciones_vs_reales.png')
#     plt.close()

#     # Loguea el gráfico en MLflow
#     mlflow.log_artifact('/tmp/predicciones_vs_reales.png')

# # Opcional: imprime el ID de la corrida
# print(f"Run ID: {run.info.run_id}")




# MAGIC ## Modelo 2 



feat_cols = [
'DayCount',
'DayCount^2',
'Tip_Bar Tradicional',
'Tip_Cervecería',
'Tip_Discoteca',
'Tip_Establecimiento de Tapeo',
'Tip_No Segmentado',
'Tip_Noche Temprana',
'Tip_Pastelería/Cafetería/Panadería',
'Tip_Restaurante',
'Tip_Restaurante de Imagen',
'Tip_Restaurante de Imagen con Tapeo',
'Orig_OFFLINE',
'Type_Porcentual',
'CouponDiscountAmt',
'CouponDiscountAmt_LAG_1',
'CouponDiscountAmt_LAG_2',
'CouponDiscountAmt_LAG_3',
'CouponDiscountAmt_LAG_4',
'CouponDiscountAmt_LAG_5',
'CouponDiscountAmt_LAG_6',
'CouponDiscountAmt_LAG_7',
'CouponDiscountAmt_LAG_8',
'CouponDiscountAmt_LAG_9',
'CouponDiscountAmt_LAG_10',
'CouponDiscountAmt^2',
'CouponDiscountAmt_LAG_1_POLY_2',
'CouponDiscountAmt_LAG_2_POLY_2',
'CouponDiscountAmt_LAG_3_POLY_2',
'CouponDiscountAmt_LAG_4_POLY_2',
'CouponDiscountAmt_LAG_5_POLY_2',
'CouponDiscountAmt_LAG_6_POLY_2',
'CouponDiscountAmt_LAG_7_POLY_2',
'CouponDiscountAmt_LAG_8_POLY_2',
'CouponDiscountAmt_LAG_9_POLY_2',
'CouponDiscountAmt_LAG_10_POLY_2',
'Sellout_LAG_1',
'Sellout_LAG_2',
'Sellout_LAG_3',
'Sellout_LAG_4',
'Sellout_LAG_5',
'Sellout_LAG_6',
'Sellout_LAG_7',
'Sellout_LAG_8',
'Sellout_LAG_9',
'Sellout_LAG_10',
'Sellout_LAG_1_POLY_2',
'Sellout_LAG_2_POLY_2', 
'Sellout_LAG_3_POLY_2', 
'Sellout_LAG_4_POLY_2',
'Sellout_LAG_5_POLY_2', 
'Sellout_LAG_6_POLY_2', 
'Sellout_LAG_7_POLY_2',
'Sellout_LAG_8_POLY_2', 
'Sellout_LAG_9_POLY_2', 
'Sellout_LAG_10_POLY_2',
]
target_col = 'Sellout'
features_2 = df[feat_cols]
target = df[target_col]




def imprimir_nombres_caracteristicas(feat_cols):
    for indice, caracteristica in enumerate(feat_cols):
        print(f"Índice {indice}: {caracteristica}")

# Llamada a la función con la lista feat_cols
imprimir_nombres_caracteristicas(feat_cols)




# FIT modelo CON PESOS
model_with_weight = BayesianRidge(fit_intercept=False)
model_with_weight.fit(features_2, target, sample_weight=sample_weight)
coef_with_weight = model_with_weight.coef_





# MAGIC ### Modelo Target REFERENCIAS



feat_cols_refs = [
'season_Otoño',
'season_Primavera',
'season_Verano',
'DayCount',
'DayCount^2',
'Tip_Bar Tradicional',
'Tip_Cervecería',
'Tip_Discoteca',
'Tip_Establecimiento de Tapeo',
'Tip_No Segmentado',
'Tip_Noche Temprana',
'Tip_Pastelería/Cafetería/Panadería',
'Tip_Restaurante',
'Tip_Restaurante de Imagen',
'Tip_Restaurante de Imagen con Tapeo',
'Orig_OFFLINE',
'Type_Porcentual',
'CouponDiscountAmt',
'CouponDiscountAmt_LAG_1',
'CouponDiscountAmt_LAG_2',
'CouponDiscountAmt_LAG_3',
'CouponDiscountAmt_LAG_4',
'CouponDiscountAmt_LAG_5',
'CouponDiscountAmt_LAG_6',
'CouponDiscountAmt_LAG_7',
'CouponDiscountAmt_LAG_8',
'CouponDiscountAmt_LAG_9',
'CouponDiscountAmt_LAG_10',
'CouponDiscountAmt^2',
'CouponDiscountAmt_LAG_1_POLY_2',
'CouponDiscountAmt_LAG_2_POLY_2',
'CouponDiscountAmt_LAG_3_POLY_2',
'CouponDiscountAmt_LAG_4_POLY_2',
'CouponDiscountAmt_LAG_5_POLY_2',
'CouponDiscountAmt_LAG_6_POLY_2',
'CouponDiscountAmt_LAG_7_POLY_2',
'CouponDiscountAmt_LAG_8_POLY_2',
'CouponDiscountAmt_LAG_9_POLY_2',
'CouponDiscountAmt_LAG_10_POLY_2',
'skus_uniques_LAG_1',
'skus_uniques_LAG_2',
'skus_uniques_LAG_3',
'skus_uniques_LAG_4',
'skus_uniques_LAG_5',
'skus_uniques_LAG_6',
'skus_uniques_LAG_7',
'skus_uniques_LAG_8',
'skus_uniques_LAG_9',
'skus_uniques_LAG_10',
'skus_uniques_LAG_1_POLY_2',
'skus_uniques_LAG_2_POLY_2',
'skus_uniques_LAG_3_POLY_2',
'skus_uniques_LAG_4_POLY_2',
'skus_uniques_LAG_5_POLY_2',
'skus_uniques_LAG_6_POLY_2',
'skus_uniques_LAG_7_POLY_2',
'skus_uniques_LAG_8_POLY_2',
'skus_uniques_LAG_9_POLY_2',
'skus_uniques_LAG_10_POLY_2',

]
target_col = 'skus_uniques'
features_ref = df[feat_cols_refs]
target_ref = df[target_col]




model_with_weight = BayesianRidge(fit_intercept=False)
model_with_weight.fit(features_ref, target_ref, sample_weight=sample_weight)
coef_with_weight = model_with_weight.coef_



import mlflow
logged_model_refs = 'runs:/a782e26dea69422b9e05b63fe6069439/model'

# Load model as a PyFuncModel.
loaded_model_refs = mlflow.sklearn.load_model(logged_model_refs)
loaded_model_pyfunc_refs = mlflow.pyfunc.load_model(logged_model_refs)





import pandas as pd
predictions_refs= loaded_model_pyfunc_refs.predict(pd.DataFrame(df))



y_real_refs=df['skus_uniques'].values



plt.figure(figsize=(10, 6))

plt.scatter(y_real_refs, predictions_refs, color='skyblue', label='Datos')
plt.plot([min(y_real_refs), max(y_real_refs)], [min(y_real_refs), max(y_real_refs)], color='red', linestyle='--', label='Ideal')

plt.xlabel('Valores Reales')
plt.ylabel('Valores Predichos')
plt.title('Predicciones vs Reales')
plt.legend(loc='upper left')

plt.show()

# MAGIC ### Modelo Target UNIDADES PEDIDO
df[df['Name']=='BAR BARROJA']

feat_cols_amount= [
'season_Otoño',
'season_Primavera',
'season_Verano',
'DayCount',
'DayCount^2',
'pctg_coupon_used',
'Tip_Bar Tradicional',
'Tip_Cervecería',
'Tip_Discoteca',
'Tip_Establecimiento de Tapeo',
'Tip_No Segmentado',
'Tip_Noche Temprana',
'Tip_Pastelería/Cafetería/Panadería',
'Tip_Restaurante',
'Tip_Restaurante de Imagen',
'Tip_Restaurante de Imagen con Tapeo',
'Orig_OFFLINE',
'Type_Porcentual',
'CouponDiscountAmt',
'CouponDiscountAmt_LAG_1',
'CouponDiscountAmt_LAG_2',
'CouponDiscountAmt_LAG_3',
'CouponDiscountAmt_LAG_4',
'CouponDiscountAmt_LAG_5',
'CouponDiscountAmt_LAG_6',
'CouponDiscountAmt_LAG_7',
'CouponDiscountAmt_LAG_8',
'CouponDiscountAmt_LAG_9',
'CouponDiscountAmt_LAG_10',
'CouponDiscountAmt^2',
'CouponDiscountAmt_LAG_1_POLY_2',
'CouponDiscountAmt_LAG_2_POLY_2',
'CouponDiscountAmt_LAG_3_POLY_2',
'CouponDiscountAmt_LAG_4_POLY_2',
'CouponDiscountAmt_LAG_5_POLY_2',
'CouponDiscountAmt_LAG_6_POLY_2',
'CouponDiscountAmt_LAG_7_POLY_2',
'CouponDiscountAmt_LAG_8_POLY_2',
'CouponDiscountAmt_LAG_9_POLY_2',
'CouponDiscountAmt_LAG_10_POLY_2',
'Amount_LAG_1',
'Amount_LAG_2',
'Amount_LAG_3',
'Amount_LAG_4',
'Amount_LAG_5',
'Amount_LAG_6',
'Amount_LAG_7',
'Amount_LAG_8',
'Amount_LAG_9',
'Amount_LAG_10',
'Amount_LAG_1_POLY_2',
'Amount_LAG_2_POLY_2',
'Amount_LAG_3_POLY_2',
'Amount_LAG_4_POLY_2',
'Amount_LAG_5_POLY_2',
'Amount_LAG_6_POLY_2',
'Amount_LAG_7_POLY_2',
'Amount_LAG_8_POLY_2',
'Amount_LAG_9_POLY_2',
'Amount_LAG_10_POLY_2',

]
target_col = 'Amount'
features_amount = df[feat_cols_amount]
target_amount = df[target_col]

model_with_weight = BayesianRidge(fit_intercept=False)
model_with_weight.fit(features_amount, target_amount, sample_weight=sample_weight)
coef_with_weight = model_with_weight.coef_
