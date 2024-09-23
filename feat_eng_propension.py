# Importar librerías
import pandas as pd
import numpy as np



# Leer datos
maestro_cliente_path = r'C:\Users\ctrujils\Downloads\maestro_cliente_final_12UM.parquet'
maestro_clientes = pd.read_parquet(maestro_cliente_path)

path = r'C:\Users\ctrujils\Downloads\order_detail.parquet'
order_detail = pd.read_parquet(path)


# print(order_detail.head())
# Filtrar devoluciones
order_detail = order_detail[order_detail['TotalOrderQuantity'] > 0]

# Feature Engineering
order_detail['CouponCode'] = order_detail['CouponCode'].replace('', 0).fillna(0)
order_detail['CouponDiscountPct'] = order_detail['CouponDiscountPct'].replace([None, ''], 0).astype(float)
order_detail['CouponDescription'] = order_detail['CouponDescription'].replace([None, ''], 'NoCupon')
order_detail['CouponDiscountAmt'] = pd.to_numeric(order_detail['CouponDiscountAmt'], errors='coerce')

# Seleccionar columnas relevantes
order_detail_sorted_filtrado_selected = order_detail[['Code', 'DistributorID', 'CodeDistributor', 'Name', 'NameDistributor', 
                                     'PointOfSaleId', 'CodeProduct', 'CouponId', 'CouponCode',
                                     'CouponDescription', 'OrderDate', 'CouponDiscountAmt', 
                                     'TotalOrderPrice', 'CouponDiscountPct', 'InsertionOrigin']]

# Agrupar y combinar datos
order_detail_grouped = order_detail_sorted_filtrado_selected.groupby(['Code', 'DistributorID', 'OrderDate']).agg({
    'CodeProduct': lambda x: list(x)
}).reset_index()

columns_to_keep = ['Code', 'OrderDate', 'Name', 'NameDistributor', 'DistributorID', 'TotalOrderPrice', 
                    'CouponCode', 'PointOfSaleId', 'CouponDiscountAmt', 'CouponDiscountPct', 
                    'CouponDescription', 'InsertionOrigin']
other_columns = order_detail_sorted_filtrado_selected[columns_to_keep].drop_duplicates()
order_detail_final = pd.merge(order_detail_grouped, other_columns, on=['Code', 'DistributorID', 'OrderDate'], how='left')
order_detail_sorted = order_detail_final.sort_values(by=['Code', 'OrderDate', 'Name'])

# Renombrar columna
order_detail_sorted = order_detail_sorted.rename(columns={'TotalOrderPrice': 'Sellout'})

# Filtrar datos por tipo de cupón
pedidos_con_cupon = order_detail_sorted[order_detail_sorted['CouponDescription'] != 'NoCupon']['PointOfSaleId'].value_counts()
pedidos_sin_cupon = order_detail_sorted[order_detail_sorted['CouponDescription'] == 'NoCupon']['PointOfSaleId'].value_counts()

# # Análisis de datos
# order_detail_sorted_filtered_con_cupon = order_detail_sorted[
#     (order_detail_sorted['CouponDiscountAmt'] > 0) | (order_detail_sorted['CouponDiscountPct'] > 0)]
# pdvs_con_cupon = order_detail_sorted_filtered_con_cupon['PointOfSaleId'].unique()

# order_detail_sorted_filtered_sin_cupon = order_detail_sorted[
#     (order_detail_sorted['CouponDiscountAmt'] == 0.0) & (order_detail_sorted['CouponDiscountPct'] == 0.0)]
# pdvs_sin_cupon = order_detail_sorted_filtered_sin_cupon['PointOfSaleId'].unique()

# duplicados_entre_grupos = set(pdvs_con_cupon).intersection(set(pdvs_sin_cupon))
# pdvs_nunca_cupon = set(pdvs_sin_cupon) - set(pdvs_con_cupon)


# Datos online
path = r'C:\Users\ctrujils\Downloads\pedidos_coincidentes.parquet'
pedidos_online=pd.read_parquet(path)


pedidos_online['OrderDate'] = pd.to_datetime(pedidos_online['OrderDate'])

def classification_origin(row):
    if pd.notna(row['InsertionOrigin']):
        if row['InsertionOrigin'] in ['RENTABILIB', 'eComm RENTABILIB']:
            return 'ONLINE'
        else:
            return 'OFFLINE'
    return 'OFFLINE'

order_detail_sorted['Origin'] = order_detail_sorted.apply(classification_origin, axis=1)

maestro_clientes = maestro_clientes[['tipologia', 'company_id']]
order_detail_sorted = pd.merge(order_detail_sorted, maestro_clientes, left_on='PointOfSaleId', right_on='company_id', how='left')
order_detail_sorted.drop(columns='company_id', inplace=True)
order_detail_sorted.dropna(subset=['tipologia', 'Name'], inplace=True)

# Limpieza de datos
order_detail_sorted = order_detail_sorted[['Code', 'OrderDate', 'Sellout','CodeProduct','Name', 'NameDistributor', 
                                           'PointOfSaleId', 'CouponCode', 'CouponDiscountAmt', 'CouponDiscountPct', 
                                           'CouponDescription', 'InsertionOrigin', 'Origin', 'tipologia']]

order_detail_sorted['NameDistributor'] = order_detail_sorted['NameDistributor'].apply(lambda x: x.replace(' BC', '').title())

pedidos_online = pedidos_online[pedidos_online['NameDistributor'].isin(order_detail_sorted['NameDistributor'].unique())]

# Unir y ajustar datos
a = order_detail_sorted[order_detail_sorted['Origin'] == 'OFFLINE']
a['OrderDate'] = pd.to_datetime(a['OrderDate']).dt.date
order_detail_sorted = pd.concat([a, pedidos_online], axis=0)
order_detail_sorted['OrderDate'] = pd.to_datetime(order_detail_sorted['OrderDate'])
order_detail_sorted.dropna(subset=['Code', 'CodeProduct', 'Origin'], inplace=True)


# Analizar frecuencia de compra online

order_detail_online = order_detail_sorted[order_detail_sorted['Origin'] == 'Online']

frequency_online_per_pdv = order_detail_online.groupby('PointOfSaleId').size().reset_index(name='Frequency_online')
order_detail_sorted = order_detail_sorted.merge(frequency_online_per_pdv, on='PointOfSaleId', how='left')



# Crear variable temporal
start_date = order_detail_sorted['OrderDate'].min()
end_date = order_detail_sorted['OrderDate'].max() + pd.Timedelta(days=1)
date_range = pd.date_range(start=start_date, end=end_date)
date_order_detail_sorted = pd.DataFrame(date_range, columns=['OrderDate'])
date_order_detail_sorted['DayCount'] = (date_order_detail_sorted['OrderDate'] - start_date).dt.days + 1
order_detail_sorted = pd.merge_asof(order_detail_sorted.sort_values('OrderDate'), date_order_detail_sorted, on='OrderDate', direction='forward')

# Crear variable de tipo de cupón
order_detail_sorted['CouponDiscountAmt'] = order_detail_sorted['CouponDiscountAmt'].astype(float)
order_detail_sorted['CouponDiscountPct'] = pd.to_numeric(order_detail_sorted['CouponDiscountPct'], errors='coerce')

def clasificar_descuento(row):
    if pd.notna(row['CouponDiscountAmt']) and pd.notna(row['CouponDiscountPct']) and pd.notna(row['CouponCode']):
        if row['CouponDiscountAmt'] != 0 and row['CouponDiscountPct'] == 0:
            return 'Fijo'
        elif row['CouponDiscountPct'] != 0 and row['CouponCode'] != 0:
            return 'Porcentual'
    return 'NoCupon'

order_detail_sorted['Coupon_type'] = order_detail_sorted.apply(clasificar_descuento, axis=1)


# Recurrencia a la compra de cupones
# print(order_detail_sorted[order_detail_sorted["Origin"] == "OFFLINE"]['PointOfSaleId'])

order_detail_sorted['CouponCode'] = order_detail_sorted['CouponCode'].replace(['', 'None', '0', '0.0', '0.00', 'NaN', 0], np.nan)
sales_coupon = order_detail_sorted[order_detail_sorted['CouponCode'].notna()].groupby('PointOfSaleId')['Sellout'].sum()
total_sales = order_detail_sorted.groupby('PointOfSaleId')['Sellout'].sum()
pctg_coupon_used = (sales_coupon / total_sales) * 100 
pctg_coupon_used_order_detail_sorted = pctg_coupon_used.reset_index(name='PctgCouponUsed')

order_detail_sorted = order_detail_sorted.merge(pctg_coupon_used_order_detail_sorted, on='PointOfSaleId', how='left')
order_detail_sorted['PctgCouponUsed'] = order_detail_sorted['PctgCouponUsed'].fillna(0)


#Dias transcurridos desde la ultima compra online (acumulado)

order_detail_online = order_detail_online.sort_values(by=['PointOfSaleId', 'OrderDate'])
order_detail_online['DaysBetweenPurchases'] = order_detail_online.groupby('PointOfSaleId')['OrderDate'].diff().dt.days
order_detail_online['DaysBetweenPurchases'] = order_detail_online['DaysBetweenPurchases'].fillna(0)
order_detail_online['CumulativeAvgDaysBetweenPurchases'] = order_detail_online.groupby('PointOfSaleId')['DaysBetweenPurchases'].expanding().mean().reset_index(level=0, drop=True)

order_detail_online_subset = order_detail_online[['PointOfSaleId', 'OrderDate', 'DaysBetweenPurchases', 'CumulativeAvgDaysBetweenPurchases']]

# Realiza el merge con el DataFrame order_detail_sorted
order_detail_sorted = pd.merge(order_detail_sorted, 
                               order_detail_online_subset, 
                               on=['PointOfSaleId', 'OrderDate'],
                               how='left')


# Booleana de ultima compra online

order_detail_sorted = order_detail_sorted.sort_values(by=['PointOfSaleId', 'NameDistributor', 'OrderDate'])
order_detail_sorted['IsOnline'] = order_detail_sorted['Origin'] == 'Online'
order_detail_sorted['LastPurchaseOnline'] = order_detail_sorted.groupby(['PointOfSaleId', 'NameDistributor'])['IsOnline'].shift(1)
order_detail_sorted['LastPurchaseOnline'] = order_detail_sorted['LastPurchaseOnline'].fillna(False).astype(bool)


## CREACIÓN DE LA VARIABLE TARGET

gt = order_detail_sorted.groupby(['NameDistributor', 'PointOfSaleId'])
digitalizacion = []

list(gt)[0][1].IsOnline \
              .iloc[::-1].rolling(5, min_periods=0) \
                         .sum().iloc[::-1] \
                               .apply(lambda x: x > 0)
list(gt)[0][1][["OrderDate", "IsOnline"]]
list(gt)[0][1][["OrderDate", "IsOnline"]].shift(-1)
list(gt)[0][1][["OrderDate", "IsOnline"]] \
    .IsOnline \
    .shift(-1) \
    .rolling(5, min_periods=0) \
    .sum() \
    .apply(lambda x: x > 0)

# .iloc[::-1]


for _, g in gt:
    g["Digitalizacion"] = g.IsOnline.rolling(10, min_periods=1).sum()
    g["ForwardOnline"] = g.IsOnline \
                           .iloc[::-1].rolling(5, min_periods=0).sum().iloc[::-1] \
                           .apply(lambda x: x > 0)
    digitalizacion.append(g)

order_detail_sorted = pd.concat(digitalizacion)


# # # Guardar archivo final
save_path = r'C:\Users\ctrujils\order_detail_sorted.parquet'
order_detail_sorted.to_parquet(save_path, index=False)

print('fin')
# # path = '/dbfs/mnt/MSM/raw_data/order_detail_cupons.parquet'
# # save_file_to_format(order_detail_sorted, path=path, format="parquet")

