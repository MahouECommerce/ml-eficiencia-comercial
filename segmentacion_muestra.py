
import pandas as pd
import numpy as np
import itertools as it
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
# Cargar los datos

path = r'C:\Users\ctrujils\order_detail_sorted_feat_eng.parquet'
order_detail_sorted = pd.read_parquet(path)



# Columnas que ya están a nivel de PDV
cols_already_pdv_level = [
    'AverageSellout', 'DaysSinceLastPurchase', 'DaysSinceFirstPurchase', 
    'TotalOnlinePurchases', 'TotalOfflinePurchases', 'AverageSelloutOnline', 
    'AverageSelloutOffline', 'DaysSinceLastOfflinePurchase', 'DaysSinceLastOnlinePurchase','CumulativeAvgDaysBetweenPurchases','Frequency_online','FrequencyTotal',
    'NumeroReferenciasUnicas','NameDistributor','Name', 'DaysSinceFirstOnlinePurchase','frequency_online_last_90','frequency_online_12UM'
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

pdvs_online=order_detail_sorted.loc[order_detail_sorted['Origin']=='Online']['PointOfSaleId'].drop_duplicates()
clientes_solo_offline = order_detail_sorted_grouped[~order_detail_sorted_grouped['PointOfSaleId'].isin(pdvs_online)]

pdvs_offline = clientes_solo_offline['PointOfSaleId'].unique().tolist()

order_detail_sorted_grouped['segmento'] = ''

order_detail_sorted_grouped.loc[order_detail_sorted_grouped['PointOfSaleId'].isin(pdvs_offline), 'segmento'] = 'solo_offline'


# Segmento de CLIENTES NUEVOS

#En base a lo que hemos analizado en el notebook 'Analisis_Segmentacion' nuestro periodo de observación para los pdvs nuevos son 90 días y/o más de tres compras

df_online=order_detail_sorted_grouped[order_detail_sorted_grouped['PointOfSaleId'].isin(pdvs_online)]
df_online['PointOfSaleId'] = df_online['PointOfSaleId'].astype(str)

filtro_nuevos = df_online[
    (df_online['DaysSinceFirstOnlinePurchase'] <= 90) &
    (df_online['frequency_online_last_90'] < 3)
].PointOfSaleId.unique().tolist()

order_detail_sorted_grouped.loc[order_detail_sorted_grouped['PointOfSaleId'].isin(filtro_nuevos), 'segmento']= 'nuevos'




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

order_detail_sorted_grouped.loc[
    (order_detail_sorted_grouped['PointOfSaleId'].isin(pdvs_dormidos)) &
    (order_detail_sorted_grouped['segmento'] == ''),
    'segmento'
] = 'dormidos'




print('fin')

# QUITAMOS A LOS PUNTOS DE VENTA CUPONEROS

# Filtrar los clientes cuponeros: aquellos con todas sus compras online usando cupones

clientes_cuponeros = order_detail_sorted_grouped[order_detail_sorted_grouped['PctgCouponUsed'] >= 95 ]
pdvs_cuponeros = clientes_cuponeros['PointOfSaleId'].unique().tolist()


order_detail_sorted_grouped.loc[
    (order_detail_sorted_grouped['PointOfSaleId'].isin(pdvs_cuponeros)) &
    (order_detail_sorted_grouped['segmento'] == ''),
    'segmento'
] = 'cuponeros'



order_detail_sorted_grouped.loc[order_detail_sorted_grouped['segmento'] == '', 'segmento'] = 'optimos'






path=r'C:\Users\ctrujils\order_detail_grouped_nuevos_segmentos.parquet'
order_detail_sorted_grouped.to_parquet(path)

# path=r'C:\Users\ctrujils\order_detail_grouped_nuevos_segmentos.csv'
# order_detail_sorted_grouped.to_csv(path, sep=';', decimal=',', index=False , encoding='utf-8-sig')




## Selección de optimos para clusterizacion
optimos_df=order_detail_sorted_grouped[order_detail_sorted_grouped['segmento']=='optimos']
## Seleccion de variables



variables_comportamentales= optimos_df [['Frequency_online', 'PctgCouponUsed', 'CouponDiscountAmt',
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
optimos_df['Cluster'] = kmeans.labels_

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


cluster_summary = optimos_df.groupby('Cluster')[variables].agg(['mean', 'median', 'std']).reset_index()
print(cluster_summary)



# Resumen de distribución de clusters por distribuidor
cluster_por_distribuidor = optimos_df.groupby(['Cluster', 'NameDistributor'])['PointOfSaleId'].count().unstack(fill_value=0)
print(cluster_por_distribuidor)


# Perfilado 
perfil_cluster = optimos_df.groupby('Cluster')[['FrequencyTotal', 'Sellout', 'NumeroReferenciasUnicas']].agg(['mean', 'median', 'std']).reset_index()
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
    pdvs_optimos_distribuidor = optimos_df[optimos_df['NameDistributor'] == distribuidor]
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
    knn = NearestNeighbors(n_neighbors=3, metric='euclidean')
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
        
    

    })

    # Agregar los resultados del distribuidor actual a la lista
    clientes_espejo_list.append(clientes_espejo_distribuidor)

# Concatenar todos los resultados en un único DataFrame
clientes_espejo_final = pd.concat(clientes_espejo_list, ignore_index=True)


path=r'C:\Users\ctrujils\clientes_espejo.parquet'
clientes_espejo_final.to_parquet(path)

print('fin')

###########

# Análisis de importancia de características para variables KNN

feature_cols = ['Sellout', 'FrequencyTotal', 'NumeroReferenciasUnicas','DaysSinceFirstPurchase' ,'DaysSinceLastPurchase']
X = order_detail_sorted_grouped[feature_cols]
y = order_detail_sorted_grouped['segmento']

# Escalar las características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Entrenar el modelo Random Forest para estimar la importancia de las características
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_scaled, y)

# Obtener la importancia de las características
importances = rf.feature_importances_

# Crear un DataFrame para la importancia de las características
feature_importances = pd.DataFrame({'Feature': feature_cols, 'Importance': importances})
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

# # Graficar la importancia de las características
# plt.figure(figsize=(10, 6))
# plt.bar(feature_importances['Feature'], feature_importances['Importance'], color='skyblue')
# plt.xlabel('Características', fontsize=14)
# plt.ylabel('Importancia', fontsize=14)
# plt.title('Importancia de las Características (Random Forest)', fontsize=16)
# plt.xticks(rotation=45)
# plt.show()

## Las variables determinantes serían  Sellout, Dias desde ultimo pedido 
print('fin')




