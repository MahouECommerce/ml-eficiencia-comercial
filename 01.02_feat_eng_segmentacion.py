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


order_detail_sorted['OrderDate'] = pd.to_datetime(order_detail_sorted['OrderDate'])
compras_online = order_detail_sorted[order_detail_sorted['Origin'] == 'Online']
first_online_purchase_date = compras_online.groupby('PointOfSaleId')['OrderDate'].min().reset_index()
first_online_purchase_date.rename(columns={'OrderDate': 'FirstOnlinePurchaseDate'}, inplace=True)
order_detail_sorted = order_detail_sorted.merge(first_online_purchase_date, on='PointOfSaleId', how='left')

order_detail_sorted['DaysSinceFirstOnlinePurchase'] = (
    last_date - order_detail_sorted['FirstOnlinePurchaseDate']
).dt.days

### Caculamos frequency para ese periodo de adaptación

ninety_days_ago = last_date - pd.Timedelta(days=90)
compras_online_last_90 = order_detail_sorted[
    (order_detail_sorted['Origin'] == 'Online') & 
    (order_detail_sorted['OrderDate'] > ninety_days_ago)
]

frequency_online_last_90 = compras_online_last_90.groupby('PointOfSaleId').size().reset_index(name='frequency_online_last_90')
order_detail_sorted = order_detail_sorted.merge(frequency_online_last_90, on='PointOfSaleId', how='left')

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
