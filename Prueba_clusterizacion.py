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


####### DROP DE BALEARES Y CERES ( no incluidos en el análsis)
order_detail_sorted=order_detail_sorted[~order_detail_sorted['NameDistributor'].isin(['Voldis Baleares','Ceres'])]


#### CLIENTES ESPEJO #####
order_detail_sorted=order_detail_sorted.sort_values(by='OrderDate', ascending=True)

# Creación de freq total de compras
frequency_total_per_pdv = order_detail_sorted.groupby('PointOfSaleId').size().reset_index(name='FrequencyTotal')


### FEATURE ENG
order_detail_sorted['CodeProduct'] = order_detail_sorted['CodeProduct'].apply(lambda x: x[0] if isinstance(x, (list, np.ndarray)) else x)

numero_referencias_unicas = order_detail_sorted.groupby('PointOfSaleId')['CodeProduct'].nunique().reset_index(name='NumeroReferenciasUnicas')


new_features=pd.merge(frequency_total_per_pdv,numero_referencias_unicas, on='PointOfSaleId' ,how='left')
order_detail_sorted=order_detail_sorted.merge(new_features, on='PointOfSaleId', how='left')

order_detail_sorted['OrderDate'] = pd.to_datetime(order_detail_sorted['OrderDate'])


order_detail_sorted['OrderDate'] = pd.to_datetime(order_detail_sorted['OrderDate'])

# Average Sellout per PointOfSaleId
order_detail_sorted['AverageSellout'] = order_detail_sorted.groupby('PointOfSaleId')['Sellout'].transform('mean')

# 
order_detail_sorted['DaysSinceLastPurchase'] = (pd.Timestamp.now() - order_detail_sorted.groupby('PointOfSaleId')['OrderDate'].transform('max')).dt.days

#
order_detail_sorted['DaysSinceFirstPurchase'] = (pd.Timestamp.now() - order_detail_sorted.groupby('PointOfSaleId')['OrderDate'].transform('min')).dt.days

# 
order_detail_sorted['TotalOnlinePurchases'] = order_detail_sorted[order_detail_sorted['Origin'] == 'Online'].groupby('PointOfSaleId')['Origin'].transform('count')
order_detail_sorted['TotalOfflinePurchases'] = order_detail_sorted[order_detail_sorted['Origin'] != 'Online'].groupby('PointOfSaleId')['Origin'].transform('count')

# 
order_detail_sorted['TotalOnlinePurchases'] = order_detail_sorted['TotalOnlinePurchases'].fillna(0)
order_detail_sorted['TotalOfflinePurchases'] = order_detail_sorted['TotalOfflinePurchases'].fillna(0)

# 
order_detail_sorted['AverageSelloutOnline'] = order_detail_sorted[order_detail_sorted['Origin'] == 'Online'].groupby('PointOfSaleId')['Sellout'].transform('mean')
order_detail_sorted['AverageSelloutOffline'] = order_detail_sorted[order_detail_sorted['Origin'] != 'Online'].groupby('PointOfSaleId')['Sellout'].transform('mean')

# 
order_detail_sorted['AverageSelloutOnline'] = order_detail_sorted['AverageSelloutOnline'].fillna(0)
order_detail_sorted['AverageSelloutOffline'] = order_detail_sorted['AverageSelloutOffline'].fillna(0)

# 
order_detail_sorted['DaysSinceLastOfflinePurchase'] = (pd.Timestamp.now() - order_detail_sorted[order_detail_sorted['Origin'] != 'Online'].groupby('PointOfSaleId')['OrderDate'].transform('max')).dt.days
order_detail_sorted['DaysSinceLastOnlinePurchase'] = (pd.Timestamp.now() - order_detail_sorted[order_detail_sorted['Origin'] == 'Online'].groupby('PointOfSaleId')['OrderDate'].transform('max')).dt.days

# 
order_detail_sorted['DaysSinceLastOfflinePurchase'] = order_detail_sorted['DaysSinceLastOfflinePurchase'].fillna(0)
order_detail_sorted['DaysSinceLastOnlinePurchase'] = order_detail_sorted['DaysSinceLastOnlinePurchase'].fillna(0)


#### Agrupamos el Df por PDV, Distribuidor
order_detail_sorted= order_detail_sorted.drop_duplicates(subset=['PointOfSaleId', 'NameDistributor']).reset_index(drop=True)

print('fin')

###### Segmento de CLIENTES OFFLINE
# Filtrar los clientes que tienen todas sus compras offline
clientes_solo_offline = order_detail_sorted.groupby('PointOfSaleId').filter(lambda x: (x['Origin'] != 'Online').all())

pdvs_offline=clientes_solo_offline['PointOfSaleId'].unique().tolist()

order_detail_sorted = order_detail_sorted[~order_detail_sorted['PointOfSaleId'].isin(pdvs_offline)]

###### Segmento de CLIENTES DORMIDOS
#Definimos corte en base a la distribución
dias_corte = 150

# Obtener la última fecha de compra online para cada PDV y calcular cuántos días han pasado desde entonces
last_dates = order_detail_sorted['OrderDate'].max()
online_orders = order_detail_sorted[order_detail_sorted['Origin'] == 'Online']
last_online_order_dates = online_orders.groupby(['PointOfSaleId'])['OrderDate'].max().reset_index()
last_online_order_dates.columns = ['PointOfSaleId', 'LastOnlineOrderDate']
last_online_order_dates['DaysSinceLastOnline'] = (last_dates - last_online_order_dates['LastOnlineOrderDate']).dt.days

# Filtrar los clientes dormidos
clientes_dormidos = last_online_order_dates[last_online_order_dates['DaysSinceLastOnline'] > dias_corte]

pdvs_dormidos = clientes_dormidos['PointOfSaleId'].unique().tolist()

order_detail_sorted = order_detail_sorted[~order_detail_sorted['PointOfSaleId'].isin(pdvs_dormidos)]



print('fin')

######## QUITAMOS A LOS PUNTOS DE VENTA CUPONEROS

# Filtrar los clientes cuponeros: aquellos con todas sus compras online usando cupones

clientes_cuponeros=order_detail_sorted[order_detail_sorted['PctgCouponUsed']==100]
pdvs_cuponeros = clientes_cuponeros['PointOfSaleId'].unique().tolist()
order_detail_sorted = order_detail_sorted[~order_detail_sorted['PointOfSaleId'].isin(pdvs_cuponeros)]


print('fin')








# ## Seleccion de variables


# variables_comportamentales= order_detail_sorted [['Frequency_online', 'PctgCouponUsed', 'CouponDiscountAmt',
#                 'Sellout', 'AverageSellout', 'DaysSinceLastPurchase',
#                 'DaysSinceFirstPurchase', 'TotalOnlinePurchases',# SACAR PORCENTAJE
#                 'TotalOfflinePurchases', #SACAR PORCENTAJE
#                  'AverageSelloutOnline',
#                 'AverageSelloutOffline', 'DaysSinceLastOfflinePurchase',
#                 'DaysSinceLastOnlinePurchase',
#                  'CumulativeAvgDaysBetweenPurchases','FrequencyTotal', 'NumeroReferenciasUnicas']]

# variables_comportamentales = variables_comportamentales.fillna(0)

# # Estandarizar las variables
# scaler = MinMaxScaler()
# scaled_features = scaler.fit_transform(variables_comportamentales)

# # Método del codo para determinar el número de clusters
# inercia = []
# range_clusters = range(1, 10)
# for k in range_clusters:
#     kmeans = KMeans(n_clusters=k, init= 'k-means++' ,random_state=42)
#     kmeans.fit(scaled_features)
#     inercia.append(kmeans.inertia_)

# # Graficar el método del codo
# plt.figure(figsize=(8, 5))
# plt.plot(range_clusters, inercia, marker='o')
# plt.xlabel('Número de Clusters')
# plt.ylabel('Inercia')
# plt.title('Método del Codo para Seleccionar el Número de Clusters')
# # plt.show()


# #El método del codo indica seleccionar entre 7 y 8 clusters
# kmeans = KMeans(n_clusters=5, init= 'k-means++', random_state=42)
# kmeans.fit(scaled_features)
# order_detail_sorted['Cluster'] = kmeans.labels_

# # Evaluar el score de Silhouette
# from sklearn.metrics import silhouette_score
# silhouette_avg = silhouette_score(scaled_features, kmeans.labels_)
# print(f"Silhouette Score: {silhouette_avg}")



# #Análisis de Cluster
# variables=['Frequency_online', 'PctgCouponUsed', 'CouponDiscountAmt',
#                 'Sellout', 'AverageSellout', 'DaysSinceLastPurchase',
#                 'DaysSinceFirstPurchase', 'TotalOnlinePurchases',
#                 'TotalOfflinePurchases', 'AverageSelloutOnline',
#                 'AverageSelloutOffline', 'DaysSinceLastOfflinePurchase',
#                 'DaysSinceLastOnlinePurchase',
#                 'CumulativeAvgDaysBetweenPurchases','FrequencyTotal', 'NumeroReferenciasUnicas', 'Digitalizacion']


# cluster_summary = order_detail_sorted.groupby('Cluster')[variables].agg(['mean', 'median', 'std']).reset_index()
# print(cluster_summary)



# # Ordenar los PDVs dentro de cada cluster según el nivel de digitalización (descendente)
# pdvs_mas_digitalizados = order_detail_sorted.sort_values(by=['Cluster', 'Digitalizacion'], ascending=[True, False])

# # # Ver los PDVs más digitalizados de cada cluster
# # print(pdvs_mas_digitalizados[['PointOfSaleId', 'Cluster', 'Digitalizacion']].head(10))



# path= r'C:\Users\ctrujils\order_detail_sorted_clusterizado.csv'
# order_detail_sorted.to_csv(path)



# print('fin')













