import pandas as pd
import numpy as np
import itertools as it
from sklearn.linear_model import BayesianRidge, ARDRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
# Cargar los datos

path = r'C:\Users\ctrujils\order_detail_sorted_normalizado.parquet'
order_detail_sorted = pd.read_parquet(path)


# DROP DE BALEARES Y CERES ( no incluidos en el análsis)
order_detail_sorted = order_detail_sorted[~order_detail_sorted['NameDistributor'].isin([
                                                                                       'Voldis Baleares', 'Ceres'])]

# FEATURE ENG

order_detail_sorted = order_detail_sorted.sort_values(
    by='OrderDate', ascending=True)

# Creación de freq total de compras
frequency_total_per_pdv = order_detail_sorted.groupby(
    'PointOfSaleId').size().reset_index(name='FrequencyTotal')



order_detail_sorted['CodeProduct'] = order_detail_sorted['CodeProduct'].apply(
    lambda x: x[0] if isinstance(x, (list, np.ndarray)) else x)

numero_referencias_unicas = order_detail_sorted.groupby('PointOfSaleId')[
    'CodeProduct'].nunique().reset_index(name='NumeroReferenciasUnicas')


new_features = pd.merge(frequency_total_per_pdv,
                        numero_referencias_unicas, on='PointOfSaleId', how='left')
order_detail_sorted = order_detail_sorted.merge(
    new_features, on='PointOfSaleId', how='left')

order_detail_sorted['OrderDate'] = pd.to_datetime(
    order_detail_sorted['OrderDate'])


order_detail_sorted['OrderDate'] = pd.to_datetime(
    order_detail_sorted['OrderDate'])

# Average Sellout per PointOfSaleId
order_detail_sorted['AverageSellout'] = order_detail_sorted.groupby(
    'PointOfSaleId')['Sellout'].transform('mean')

#
last_date = order_detail_sorted.OrderDate.max()
order_detail_sorted['DaysSinceLastPurchase'] = (
    last_date
    - order_detail_sorted.groupby('PointOfSaleId')['OrderDate'].transform('max')).dt.days

#
order_detail_sorted['DaysSinceFirstPurchase'] = (
    last_date
    - order_detail_sorted.groupby('PointOfSaleId')['OrderDate'].transform('min')).dt.days


total_online_purchases = order_detail_sorted[order_detail_sorted['IsOnline'] == True].groupby('PointOfSaleId').size().reset_index(name='TotalOnlinePurchases')
total_offline_purchases = order_detail_sorted[order_detail_sorted['IsOnline'] == False].groupby('PointOfSaleId').size().reset_index(name='TotalOfflinePurchases')
order_detail_sorted = order_detail_sorted.merge(total_online_purchases, on='PointOfSaleId', how='left')
order_detail_sorted = order_detail_sorted.merge(total_offline_purchases, on='PointOfSaleId', how='left')
order_detail_sorted['TotalOnlinePurchases'] = order_detail_sorted['TotalOnlinePurchases'].fillna(0)
order_detail_sorted['TotalOfflinePurchases'] = order_detail_sorted['TotalOfflinePurchases'].fillna(0)


#
order_detail_sorted['AverageSelloutOnline'] = order_detail_sorted[order_detail_sorted['IsOnline']
                                                                  == True].groupby('PointOfSaleId')['Sellout'].transform('mean')
order_detail_sorted['AverageSelloutOffline'] = order_detail_sorted[order_detail_sorted['IsOnline']
                                                                   == False].groupby('PointOfSaleId')['Sellout'].transform('mean')

#
order_detail_sorted['AverageSelloutOnline'] = order_detail_sorted['AverageSelloutOnline'].fillna(
    0)
order_detail_sorted['AverageSelloutOffline'] = order_detail_sorted['AverageSelloutOffline'].fillna(
    0)


######### Análisis de cortes y ratios para determinar temporalidad de adopcionen la plataforma

df_online = order_detail_sorted[order_detail_sorted['IsOnline'] == True]
df_online = df_online.sort_values(by=['PointOfSaleId', 'OrderDate'])

df_online_unique_dates = df_online.drop_duplicates(subset=['PointOfSaleId', 'OrderDate'], keep='first')

df_online_unique_dates['DaysBetweenOnlinePurchases'] = df_online_unique_dates.groupby('PointOfSaleId')['OrderDate'].diff().dt.days


def calculate_days_between_purchases(group):
    group = group.sort_values(by='OrderDate')
    days_between = group['OrderDate'].diff().dt.days
    group['DaysBetweenFirstAndSecond'] = days_between.shift(-1) if len(days_between) > 1 else None
    group['DaysBetweenSecondAndThird'] = days_between.shift(-2) if len(days_between) > 2 else None
    return group


df_online_unique_dates = df_online_unique_dates.groupby('PointOfSaleId').apply(calculate_days_between_purchases)


days_between_dict = df_online_unique_dates.set_index(['PointOfSaleId', 'OrderDate'])[['DaysBetweenOnlinePurchases', 'DaysBetweenFirstAndSecond', 'DaysBetweenSecondAndThird']].to_dict(orient='index')

order_detail_sorted['DaysBetweenOnlinePurchases'] = order_detail_sorted.set_index(['PointOfSaleId', 'OrderDate']).index.map(lambda x: days_between_dict.get(x, {}).get('DaysBetweenOnlinePurchases', 0))
order_detail_sorted['DaysBetweenFirstAndSecond'] = order_detail_sorted.set_index(['PointOfSaleId', 'OrderDate']).index.map(lambda x: days_between_dict.get(x, {}).get('DaysBetweenFirstAndSecond', 0))
order_detail_sorted['DaysBetweenSecondAndThird'] = order_detail_sorted.set_index(['PointOfSaleId', 'OrderDate']).index.map(lambda x: days_between_dict.get(x, {}).get('DaysBetweenSecondAndThird', 0))

# Llenar NaN con ceros
order_detail_sorted.fillna(0, inplace=True)





# Verificar si hay valores NaN
print(order_detail_sorted['DaysBetweenOnlinePurchases'].isna().sum())

# Mostrar ejemplos específicos para verificar valores
print(df_online_unique_dates[df_online_unique_dates['PointOfSaleId'] == 'CLI0020281'][['OrderDate', 'DaysBetweenOnlinePurchases']])


# Obtener la última fecha de compra en el DataFrame general
last_date = order_detail_sorted['OrderDate'].max()

# Calcular la última fecha de compra online para cada PointOfSaleId
last_online_order_dates = (
    order_detail_sorted[order_detail_sorted['Origin'] == 'Online']
    .groupby('PointOfSaleId')['OrderDate']
    .max()
    .reset_index()
)
last_online_order_dates.columns = ['PointOfSaleId', 'LastOnlineOrderDate']

# Calcular la última fecha de compra offline para cada PointOfSaleId
last_offline_order_dates = (
    order_detail_sorted[order_detail_sorted['Origin'] != 'Online']
    .groupby('PointOfSaleId')['OrderDate']
    .max()
    .reset_index()
)
last_offline_order_dates.columns = ['PointOfSaleId', 'LastOfflineOrderDate']

# Combinar las fechas de compra online y offline con el DataFrame principal
order_detail_sorted = order_detail_sorted.merge(last_online_order_dates, on='PointOfSaleId', how='left')
order_detail_sorted = order_detail_sorted.merge(last_offline_order_dates, on='PointOfSaleId', how='left')

# Calcular los días desde la última compra online
order_detail_sorted['DaysSinceLastOnlinePurchase'] = (
    last_date - order_detail_sorted['LastOnlineOrderDate']
).dt.days

# Calcular los días desde la última compra offline
order_detail_sorted['DaysSinceLastOfflinePurchase'] = (
    last_date - order_detail_sorted['LastOfflineOrderDate']
).dt.days

def convert_object_to_string(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str)
    return df

order_detail_sorted = convert_object_to_string(order_detail_sorted)

path = r'C:\Users\ctrujils\order_detail_sorted_feat_eng.parquet'
order_detail_sorted.to_parquet(path)


# Columnas que ya están a nivel de PDV
cols_already_pdv_level = [
    'AverageSellout', 'DaysSinceLastPurchase', 'DaysSinceFirstPurchase', 
    'TotalOnlinePurchases', 'TotalOfflinePurchases', 'AverageSelloutOnline', 
    'AverageSelloutOffline', 'DaysSinceLastOfflinePurchase', 'DaysSinceLastOnlinePurchase','CumulativeAvgDaysBetweenPurchases','Frequency_online','FrequencyTotal',
    'NumeroReferenciasUnicas','NameDistributor','Name'
]

# Columnas que requieren agregación (a nivel de pedido)
cols_to_aggregate = {
    'PctgCouponUsed': 'mean',
    'DaysBetweenOnlinePurchases': 'mean',
    'CouponDiscountAmt': 'sum', 
    'DaysBetweenFirstAndSecond': 'mean',
    'DaysBetweenSecondAndThird': 'mean',
    'Sellout': 'sum',  
    'Digitalizacion': 'mean'  
}


# Hacer un groupby para agregar solo las columnas que lo necesitan
agg_df = order_detail_sorted.groupby('PointOfSaleId').agg(cols_to_aggregate).reset_index()

# Conservar las columnas que ya están a nivel de PDV (sin recalcular)
pdv_level_data = order_detail_sorted.drop_duplicates(subset=['PointOfSaleId'])[cols_already_pdv_level + ['PointOfSaleId']]

# Hacer el merge de las columnas agregadas con las que ya están a nivel de PDV
order_detail_sorted_grouped = pd.merge(pdv_level_data, agg_df, on='PointOfSaleId', how='left')

# Ver el DataFrame final con los datos correctamente agregados a nivel de PDV
print(order_detail_sorted_grouped)

path=r'C:\Users\ctrujils\order_detail_sorted_grouped.parquet'
order_detail_sorted_grouped.to_parquet(path)
print('fin')

# Segmento de CLIENTES OFFLINE
# Filtrar los clientes que tienen todas sus compras offline

online_df=order_detail_sorted.loc[order_detail_sorted['Origin']=='Online']['PointOfSaleId'].drop_duplicates()
clientes_solo_offline = order_detail_sorted_grouped[~order_detail_sorted_grouped['PointOfSaleId'].isin(online_df)]

pdvs_offline = clientes_solo_offline['PointOfSaleId'].unique().tolist()

order_detail_sorted_grouped = order_detail_sorted_grouped[~order_detail_sorted_grouped['PointOfSaleId'].isin(
    pdvs_offline)]






path= r'C:\Users\ctrujils\clientes_solo_offline.csv'
clientes_solo_offline.to_csv(path,
          sep=';',  # Delimitador
          decimal=',',  
          index=False,  
          float_format='%.2f',  # Formato para los valores decimales
)


path=r'C:\Users\ctrujils\order_detail_grouped_onine.parquet'
order_detail_sorted_grouped.to_parquet(path)


# Segmento de CLIENTES DORMIDOS
# Definimos corte en base a la distribución
dias_corte = 50

# Obtener la última fecha de compra online para cada PDV y calcular cuántos días han pasado desde entonces

last_online_order_dates = \
    order_detail_sorted_grouped[['PointOfSaleId', "DaysSinceLastOnlinePurchase"]]

# Filtrar los clientes dormidos
clientes_dormidos = \
    last_online_order_dates[
        last_online_order_dates.DaysSinceLastOnlinePurchase > dias_corte]

pdvs_dormidos = clientes_dormidos['PointOfSaleId'].unique().tolist()

order_detail_sorted_dormidos = order_detail_sorted_grouped[order_detail_sorted_grouped['PointOfSaleId'].isin(pdvs_dormidos)]



path= r'C:\Users\ctrujils\clientes_dormidos.csv'
order_detail_sorted_dormidos.to_csv(path,
          sep=';',  # Delimitador
          decimal=',',  
          index=False,  
          float_format='%.2f')  # Formato para los valores decimales 

order_detail_sorted_grouped = order_detail_sorted_grouped[~order_detail_sorted_grouped['PointOfSaleId'].isin(
    pdvs_dormidos)]



print('fin')

# QUITAMOS A LOS PUNTOS DE VENTA CUPONEROS

# Filtrar los clientes cuponeros: aquellos con todas sus compras online usando cupones

clientes_cuponeros = order_detail_sorted_grouped[order_detail_sorted_grouped['PctgCouponUsed'] >= 95 ]
pdvs_cuponeros = clientes_cuponeros['PointOfSaleId'].unique().tolist()
order_detail_sorted_grouped = order_detail_sorted_grouped[~order_detail_sorted_grouped['PointOfSaleId'].isin(
    pdvs_cuponeros)]



path= r'C:\Users\ctrujils\clientes_cuponeros.csv'
clientes_cuponeros.to_csv(path,
          sep=';',  # Delimitador
          decimal=',',  
          index=False,  
          float_format='%.2f')  # Formato para los valores decimales
    


## Seleccion de variables
path=r'C:\Users\ctrujils\muestra_optima.csv'
order_detail_sorted_grouped.to_csv(path)
variables_comportamentales= order_detail_sorted_grouped [['Frequency_online', 'PctgCouponUsed', 'CouponDiscountAmt',
                'Sellout', 'AverageSellout', 'DaysSinceLastPurchase',
                'DaysSinceFirstPurchase', 'TotalOnlinePurchases',
                'TotalOfflinePurchases', 
                 'AverageSelloutOnline',
                'AverageSelloutOffline', 'DaysSinceLastOfflinePurchase',
                'DaysSinceLastOnlinePurchase',
                 'CumulativeAvgDaysBetweenPurchases','FrequencyTotal', 'NumeroReferenciasUnicas']]

variables_comportamentales = variables_comportamentales.fillna(0)

#Estandarizar las variables
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(variables_comportamentales)

# Método del codo para determinar el número de clusters
inercia = []
range_clusters = range(1, 10)
for k in range_clusters:
    kmeans = KMeans(n_clusters=k, init= 'k-means++' ,random_state=42)
    kmeans.fit(scaled_features)
    inercia.append(kmeans.inertia_)

# Graficar el método del codo
plt.figure(figsize=(8, 5))
plt.plot(range_clusters, inercia, marker='o')
plt.xlabel('Número de Clusters')
plt.ylabel('Inercia')
plt.title('Método del Codo para Seleccionar el Número de Clusters')
# plt.show()


#El método del codo indica seleccionar entre 7 y 8 clusters
kmeans = KMeans(n_clusters=5, init= 'k-means++', random_state=42)
kmeans.fit(scaled_features)
order_detail_sorted_grouped['Cluster'] = kmeans.labels_

# Evaluar el score de Silhouette
from sklearn.metrics import silhouette_score
silhouette_avg = silhouette_score(scaled_features, kmeans.labels_)
print(f"Silhouette Score: {silhouette_avg}")


#Análisis de Cluster
variables=['Frequency_online', 'PctgCouponUsed', 'CouponDiscountAmt',
                'Sellout', 'AverageSellout', 'DaysSinceLastPurchase',
                'DaysSinceFirstPurchase', 'TotalOnlinePurchases',
                'TotalOfflinePurchases', 'AverageSelloutOnline',
                'AverageSelloutOffline', 'DaysSinceLastOfflinePurchase',
                'DaysSinceLastOnlinePurchase',
                'CumulativeAvgDaysBetweenPurchases','FrequencyTotal', 'NumeroReferenciasUnicas', 'Digitalizacion']


cluster_summary = order_detail_sorted_grouped.groupby('Cluster')[variables].agg(['mean', 'median', 'std']).reset_index()
print(cluster_summary)



# Resumen de distribución de clusters por distribuidor
cluster_por_distribuidor = order_detail_sorted_grouped.groupby(['Cluster', 'NameDistributor'])['PointOfSaleId'].count().unstack(fill_value=0)
print(cluster_por_distribuidor)


# Perfilado 
perfil_cluster = order_detail_sorted_grouped.groupby('Cluster')[['FrequencyTotal', 'Sellout', 'NumeroReferenciasUnicas']].agg(['mean', 'median', 'std']).reset_index()
print(perfil_cluster)



# Definir clusters óptimos ( aquellos con alta frecuencia, ventas, y referencias)
clusters_optimos = perfil_cluster[perfil_cluster[('FrequencyTotal', 'mean')] > perfil_cluster[('FrequencyTotal', 'mean')].median()]



print('fin')



######################################################### CLIENTES ESPEJO ANALISIS #####################################

from sklearn.neighbors import NearestNeighbors

# Obtener la lista de distribuidores únicos
distribuidores_unicos = order_detail_sorted_grouped['NameDistributor'].unique()

# Crear una lista para almacenar los resultados de los clientes espejo
clientes_espejo_list = []

# Iterar sobre cada distribuidor
for distribuidor in distribuidores_unicos:
    
    # Filtrar los PDVs offline y los óptimos para el distribuidor actual
    pdvs_optimos_distribuidor = order_detail_sorted_grouped[order_detail_sorted_grouped['NameDistributor'] == distribuidor]
    pdvs_offline_distribuidor = clientes_solo_offline[clientes_solo_offline['NameDistributor'] == distribuidor]

    # Si no hay PDVs óptimos o offline para el distribuidor, pasar al siguiente
    if pdvs_optimos_distribuidor.empty or pdvs_offline_distribuidor.empty:
        continue

    # Seleccionar las variables relevantes para el análisis
    variables_comparacion = ['Sellout', 'FrequencyTotal', 'NumeroReferenciasUnicas', 'DaysSinceLastPurchase']
    pdvs_optimos_distribuidor_subset = pdvs_optimos_distribuidor[variables_comparacion]
    pdvs_offline_distribuidor_subset = pdvs_offline_distribuidor[variables_comparacion]

    # Normalizar las variables en ambas muestras
    scaler = MinMaxScaler()
    pdvs_offline_normalized = scaler.fit_transform(pdvs_offline_distribuidor_subset)
    pdvs_optimos_normalized = scaler.transform(pdvs_optimos_distribuidor_subset)

    # Aplicar KNN para encontrar clientes espejo
    knn = NearestNeighbors(n_neighbors=5, metric='euclidean')
    knn.fit(pdvs_offline_normalized)

    # Encontrar los 5 PDVs offline más similares para cada PDV óptimo
    distances, indices = knn.kneighbors(pdvs_optimos_normalized)

    # Crear un DataFrame para ver los resultados
    clientes_espejo_distribuidor = pd.DataFrame({
        'Distribuidor': distribuidor,
        'PDV_Optimo': pdvs_optimos_distribuidor['PointOfSaleId'].values,
        'ClienteEspejo_Offline_1': pdvs_offline_distribuidor.iloc[indices[:, 0]]['PointOfSaleId'].values,
        'ClienteEspejo_Offline_2': pdvs_offline_distribuidor.iloc[indices[:, 1]]['PointOfSaleId'].values,
        'ClienteEspejo_Offline_3': pdvs_offline_distribuidor.iloc[indices[:, 2]]['PointOfSaleId'].values,
        'ClienteEspejo_Offline_4': pdvs_offline_distribuidor.iloc[indices[:, 3]]['PointOfSaleId'].values,
        'ClienteEspejo_Offline_5': pdvs_offline_distribuidor.iloc[indices[:, 4]]['PointOfSaleId'].values,
    

    })

    # Agregar los resultados del distribuidor actual a la lista
    clientes_espejo_list.append(clientes_espejo_distribuidor)

# Concatenar todos los resultados en un único DataFrame
clientes_espejo_final = pd.concat(clientes_espejo_list, ignore_index=True)


path=r'C:\Users\ctrujils\clientes_espejo.parquet'
clientes_espejo_final.to_parquet(path)

print('fin')

###########

## Probamos otro método para intentar captar mas muestra offline de todos los distribuidores. 

plt.figure(figsize=(10, 6))

# Scatter plot
plt.scatter(order_detail_sorted_grouped['Sellout'], order_detail_sorted_grouped['FrequencyTotal'], alpha=0.6, color='b', edgecolors='w')

# Ajustes para los ejes para enfocar la región relevante
plt.xlim(0, 50000)  #
plt.ylim(0, 100)    # Limites

# Ajustar los ticks para intervalos más específicos
plt.xticks(range(0, 50001, 5000))  
plt.yticks(range(0, 101, 10))      

# Añadir etiquetas y título
plt.xlabel('Sellout')
plt.ylabel('Frecuencia Total de Compras')
plt.title('Relación entre Sellout y Frecuencia Total (Ajustado)')
plt.grid(True)


plt.show()

print('fin')


# Calcular los valores de corte para FrequencyTotal y Sellout (cortes)

mean_frequency_total = order_detail_sorted_grouped['FrequencyTotal'].mean()
mean_sellout = order_detail_sorted_grouped['Sellout'].mean()


# PDVS offline que cumplen estos criterios: 

pdvs_potenciales=clientes_solo_offline[(clientes_solo_offline['FrequencyTotal']>= mean_frequency_total)& 
                                       (clientes_solo_offline['Sellout']>= mean_sellout)]


##### Scatter 

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))

# Scatterplot de todos los PDVs offline
plt.scatter(clientes_solo_offline['FrequencyTotal'], clientes_solo_offline['Sellout'], alpha=0.3, label='PDVs Offline', color='gray')

# Resaltar los PDVs potenciales para la migración (corte por media)
plt.scatter(pdvs_potenciales['FrequencyTotal'], pdvs_potenciales['Sellout'], alpha=0.6, label='Potenciales (Corte Media)', color='#8B0000')

# Añadir líneas de referencia para la media
plt.axvline(x=mean_frequency_total, color='blue', linestyle='-', linewidth=1, label='Mean FrequencyTotal')
plt.axhline(y=mean_sellout, color='blue', linestyle='-', linewidth=1, label='Mean Sellout')

#
plt.xlim(0, 500)  
plt.ylim(0, 150000)  

# Ajustar los ticks para intervalos más específicos y legibles
plt.xticks(range(0, 501, 50)) 
plt.yticks(range(0, 150001, 25000))  

# Configurar el gráfico
plt.xlabel('Frequency Total')
plt.ylabel('Sellout')
plt.title('Scatterplot de Frequency Total y Sellout para Identificar PDVs Potenciales para Migración')
plt.legend()
plt.grid(True)

# Mostrar el gráfico
plt.show()

print('fin')










############################################ ANALISIS DIGITALIZACION ##############################################################


import matplotlib.pyplot as plt

# Definir el punto de corte para la digitalización (>3 considerado digitalizado)
# Distribución de Digitalización
plt.figure(figsize=(10, 6))
plt.hist(order_detail_sorted['Digitalizacion'], bins=20, color='skyblue', edgecolor='black')
plt.xlabel('Nivel de Digitalización')
plt.ylabel('Número de PDVs')
plt.title('Distribución del Nivel de Digitalización')
# plt.show()


# Definir el punto de corte para la digitalización (>3 considerado digitalizado)
punto_corte_digitalizacion = 3

order_detail_sorted_grouped['DigitalizacionNivel'] = order_detail_sorted_grouped['Digitalizacion'].apply(
    lambda x: 'Por encima' if x > punto_corte_digitalizacion else 'Por debajo'
)

digitalizacion_total = order_detail_sorted_grouped['DigitalizacionNivel'].value_counts()
print("PDVs por nivel de digitalización (Todos juntos):")
print(digitalizacion_total)


##### DATOS
# GENERAL
#Por distribuidor
digitalizacion_por_distribuidor = order_detail_sorted_grouped.groupby(['NameDistributor', 'DigitalizacionNivel']).size().reset_index(name='Count')
print("PDVs por nivel de digitalización (Por distribuidor):")
print(digitalizacion_por_distribuidor)

#Por cluster
digitalizacion_por_cluster = order_detail_sorted_grouped.groupby(['Cluster', 'DigitalizacionNivel']).size().reset_index(name='Count')
print("PDVs por nivel de digitalización (Por cluster):")
print(digitalizacion_por_cluster)



# Media de sellout de los PDVs por nivel de digitalización
sellout_por_digitalizacion = order_detail_sorted_grouped.groupby('DigitalizacionNivel')['Sellout'].mean().reset_index()
print("Media de Sellout por nivel de digitalización (Todos juntos):")
print(sellout_por_digitalizacion)


# Media de sellout de los PDVs por nivel de digitalización y distribuidor
sellout_por_distribuidor = order_detail_sorted_grouped.groupby(['NameDistributor', 'DigitalizacionNivel'])['Sellout'].mean().reset_index()
print("Media de Sellout por nivel de digitalización (Por distribuidor):")
print(sellout_por_distribuidor)


# Media de sellout de los PDVs por nivel de digitalización y cluster
sellout_por_cluster = order_detail_sorted_grouped.groupby(['Cluster', 'DigitalizacionNivel'])['Sellout'].mean().reset_index()
print("Media de Sellout por nivel de digitalización (Por cluster):")
print(sellout_por_cluster)


#### PLOTS

# # TOTAL NIVEL DIGITALIZACION
# digitalizacion_total = order_detail_sorted_grouped['DigitalizacionNivel'].value_counts()

# plt.figure(figsize=(8, 5))
# digitalizacion_total.plot(kind='bar', color=['red', 'pink'])
# plt.xlabel('Nivel de Digitalización')
# plt.ylabel('Cantidad de PDVs')
# plt.title('Cantidad de PDVs por Nivel de Digitalización (Todos Juntos)')
# plt.xticks(rotation=0)
# plt.show()



# # DIGITALIZACION POR DISTRIBUIDOR


# distribuidores = digitalizacion_por_distribuidor['NameDistributor'].unique()
# num_distribuidores = len(distribuidores)
# fig, axes = plt.subplots(nrows=(num_distribuidores + 1) // 2, ncols=2, figsize=(14, 5 * ((num_distribuidores + 1) // 2)), sharex=False)
# axes = axes.flatten()

# for i, distribuidor in enumerate(distribuidores):
#     distribuidor_data = digitalizacion_por_distribuidor[digitalizacion_por_distribuidor['NameDistributor'] == distribuidor]
#     axes[i].bar(distribuidor_data['DigitalizacionNivel'], distribuidor_data['Count'], color=['red', 'pink'])
#     axes[i].set_title(f'Digitalización - Distribuidor {distribuidor}')
#     axes[i].set_ylabel('Cantidad de PDVs')
#     axes[i].set_xlabel('Nivel de Digitalización')

#     # Rotar las etiquetas del eje X para evitar solapamiento
#     axes[i].tick_params(axis='x', rotation=30)

# for j in range(i + 1, len(axes)):
#     fig.delaxes(axes[j])

# plt.subplots_adjust(hspace=0.6, wspace=0.4)
# plt.tight_layout()
# plt.show()




# # DIGITALIZACION POR CLUSTER
# clusters = digitalizacion_por_cluster['Cluster'].unique()

# num_clusters = len(clusters)
# fig, axes = plt.subplots(nrows=(num_clusters + 1) // 2, ncols=2, figsize=(14, 5 * ((num_clusters + 1) // 2)), sharex=False)

# # Aplanar los ejes para facilitar la iteración
# axes = axes.flatten()
# for i, cluster in enumerate(clusters):
#     cluster_data = digitalizacion_por_cluster[digitalizacion_por_cluster['Cluster'] == cluster]
#     axes[i].bar(cluster_data['DigitalizacionNivel'], cluster_data['Count'], color=['red', 'pink'])
#     axes[i].set_title(f'Cantidad de PDVs por Nivel de Digitalización - Cluster {cluster}')
#     axes[i].set_ylabel('Cantidad de PDVs')
#     axes[i].set_xlabel('Nivel de Digitalización')

#     # Rotar las etiquetas del eje X para evitar solapamiento
#     axes[i].tick_params(axis='x', rotation=30)

# for j in range(i + 1, len(axes)):
#     fig.delaxes(axes[j])
# plt.subplots_adjust(hspace=0.6, wspace=0.4)
# plt.tight_layout()
# plt.show()




print('fin')


# Crear el scatterplot para visualizar la relación entre Digitalización y Sellout
plt.figure(figsize=(10, 6))
plt.scatter(order_detail_sorted_grouped['Digitalizacion'], order_detail_sorted_grouped['PctgCouponUsed'], alpha=0.6, color='b', edgecolors='w')
plt.xlabel('Nivel de Digitalización')
plt.ylabel('Uso de cupones /total compras')
plt.title('Relación entre Nivel de Digitalización y Uso de cupones para los PDVs')
plt.grid(True)
plt.show()



print('fin')


# distribuidores = sellout_por_distribuidor['NameDistributor'].unique()

# # Crear una figura con subplots para cada distribuidor
# fig, axes = plt.subplots(len(distribuidores), 1, figsize=(12, 6 * len(distribuidores)), sharex=True)

# # Iterar sobre cada distribuidor y graficar los niveles de digitalización y la media de sellout
# for i, distribuidor in enumerate(distribuidores):
#     distribuidor_data = sellout_por_distribuidor[sellout_por_distribuidor['NameDistributor'] == distribuidor]
#     axes[i].bar(distribuidor_data['DigitalizacionNivel'], distribuidor_data['Sellout'], color=['lightblue', 'orange'])
#     axes[i].set_title(f'Media de Sellout por Nivel de Digitalización - Distribuidor {distribuidor}')
#     axes[i].set_ylabel('Media de Sellout')
#     axes[i].set_xlabel('Nivel de Digitalización')

# # Ajustar el layout para mejorar la visualización
# plt.tight_layout()
# plt.show()


# # Crear gráficos de barras para la media de sellout por nivel de digitalización - Por cluster
# clusters = sellout_por_cluster['Cluster'].unique()

# # Crear una figura con subplots para cada cluster
# fig, axes = plt.subplots(len(clusters), 1, figsize=(12, 6 * len(clusters)), sharex=True)

# # Iterar sobre cada cluster y graficar los niveles de digitalización y la media de sellout
# for i, cluster in enumerate(clusters):
#     cluster_data = sellout_por_cluster[sellout_por_cluster['Cluster'] == cluster]
#     axes[i].bar(cluster_data['DigitalizacionNivel'], cluster_data['Sellout'], color=['lightgreen', 'salmon'])
#     axes[i].set_title(f'Media de Sellout por Nivel de Digitalización - Cluster {cluster}')
#     axes[i].set_ylabel('Media de Sellout')
#     axes[i].set_xlabel('Nivel de Digitalización')

# # Ajustar el layout para mejorar la visualización
# plt.tight_layout()
# plt.show()


