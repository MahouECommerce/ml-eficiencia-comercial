# Importar libreríascolumns = order_detail_sorted
import pandas as pd
import numpy as np
import pickle



# Leer datos
maestro_cliente_path = r'C:\Users\ctrujils\maestro_cliente_final_segmentado.parquet'
maestro_clientes = pd.read_parquet(maestro_cliente_path)


#### AGREGAMOS CLUSTER 1,2,3,Y 0

# with open('cluster_dict_by_company_release_1.pickle', 'rb') as file:
#     segmentacion = pickle.load(file)

# data = []

# # Iterar sobre cada clave del diccionario (los números como '1', '4', '5', etc.)
# for cluster, sub_dict in segmentacion.items():
#     # Añadir cada PointOfSaleId y su respectivo cluster como tupla (id, cluster)
#     for point_of_sale_id, cluster_value in sub_dict.items():
#         data.append((point_of_sale_id, cluster_value))

# # Convertir la lista de tuplas en un DataFrame
# order_detail_sorted_cluster = pd.DataFrame(data, columns=['company_id', 'Cluster'])


# maestro_clientes=pd.merge(maestro_clientes,order_detail_sorted_cluster, on='company_id', how='left')



# # # # Guardar archivo final
# save_path = r'C:\Users\ctrujils\maestro_clientes_12UM_cluster.parquet'
# maestro_clientes.to_parquet(save_path, index=False)





path = r'C:\Users\ctrujils\Downloads\order_detail_total.parquet'
order_detail = pd.read_parquet(path)


# print(order_detail.head())
# Filtrar devoluciones
order_detail = order_detail[order_detail['TotalOrderQuantity'] > 0]





# Feature Engineering
order_detail['CouponCode'] = order_detail['CouponCode'].replace('', 0).fillna(0)
order_detail['CouponDiscountPct'] = order_detail['CouponDiscountPct'].replace([None, ''], 0).astype(float)
order_detail['CouponDescription'] = order_detail['CouponDescription'].replace([None, ''], 'NoCupon')
order_detail['CouponDiscountAmt'] = pd.to_numeric(order_detail['CouponDiscountAmt'], errors='coerce')

order_detail['NameDistributor'] = order_detail['NameDistributor'].replace('Voldis A Coruña', 'Voldis Coruña')


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

# maestro_clientes = maestro_clientes[['tipologia', 'Cluster','abc','company_id']]
# order_detail_sorted = pd.merge(order_detail_sorted, maestro_clientes, left_on='PointOfSaleId', right_on='company_id', how='left')
# order_detail_sorted.drop(columns='company_id', inplace=True)
# order_detail_sorted['tipologia'] = order_detail_sorted['tipologia'].fillna('No Segmentado')

# Limpieza de datos
order_detail_sorted = order_detail_sorted[['Code', 'OrderDate', 'Sellout','CodeProduct','Name', 'NameDistributor', 
                                           'PointOfSaleId', 'CouponCode', 'CouponDiscountAmt', 'CouponDiscountPct', 
                                           'CouponDescription', 'InsertionOrigin', 'Origin']]

order_detail_sorted['NameDistributor'] = order_detail_sorted['NameDistributor'].apply(lambda x: x.replace(' BC', '').title())



# Unir y ajustar datos
a = order_detail_sorted[order_detail_sorted['Origin'] == 'OFFLINE']
a['OrderDate'] = pd.to_datetime(a['OrderDate']).dt.date
order_detail_sorted = pd.concat([a, pedidos_online], axis=0)
order_detail_sorted['OrderDate'] = pd.to_datetime(order_detail_sorted['OrderDate'])
order_detail_sorted.dropna(subset=['Code', 'CodeProduct', 'Origin'], inplace=True)

# Normalizar la columna de cupones

order_detail_sorted_unicos = order_detail_sorted[['CouponCode','CouponDescription']].drop_duplicates()
order_detail_sorted_unicos.head(10)

def normalize_string(s):
    if pd.isna(s):
        return s
    return s.strip().lower()

def normalize_coupon(row):
    coupon_code = normalize_string(row['CouponCode']) if isinstance(row['CouponCode'], str) else None
    coupon_desc = normalize_string(row['CouponDescription']) if isinstance(row['CouponDescription'], str) else None
    discount_pct = row['CouponDiscountPct']
    discount_amt = row['CouponDiscountAmt']

    # Regla 1: Si el campo discount_amt es Cero o NaN, se indica que no hay cupón ('No cupón').
    if pd.isna(discount_amt) or discount_amt == 0:
        return "No cupón"

    # Regla 2: Si CouponCode y CouponDescription son iguales (después de normalizarlos), se utiliza el valor de CouponCode.
    if coupon_code and coupon_code == coupon_desc:
        return row['CouponCode']  

    # Regla 3: Si solo uno de los campos (CouponCode o CouponDescription) está informado y CouponDiscountAmt es mayor a 0, se utiliza el campo informado.
    if coupon_code and not coupon_desc and discount_amt > 0:
        return row['CouponCode']
    if coupon_desc and not coupon_code and discount_amt > 0:
        return row['CouponDescription']
    
    # Regla 4: Si discount_pct coincide con alguna parte de CouponCode o CouponDescription (como string), se utiliza ese campo.
    if not pd.isna(discount_pct):
        discount_str = f"{int(discount_pct)}"  # Convertimos discount_pct a string para hacer la comparación
        if coupon_code and discount_str in coupon_code:
            return row['CouponCode']
        elif coupon_desc and discount_str in coupon_desc:
            return row['CouponDescription']

    # Regla 5: Si el valor de CouponDiscountAmt coincide con CouponCode o CouponDescription (como string), se utiliza ese campo.
    if not pd.isna(discount_amt):
        discount_amt_str = f"{int(discount_amt)}"  # Convertimos el valor de CouponDiscountAmt a string
        if coupon_code and discount_amt_str in coupon_code:
            return row['CouponCode']
        elif coupon_desc and discount_amt_str in coupon_desc:
            return row['CouponDescription']

    # Regla final: Si no se cumple ninguna regla anterior, se devuelve 'Unknown'.
    return "Unknown"

# Aplicar la normalización al dataframe
order_detail_sorted['NormalizedCoupon'] = order_detail_sorted.apply(normalize_coupon, axis=1)





# Analizar frecuencia de compra online

order_detail_online = order_detail_sorted[order_detail_sorted['Origin'] == 'Online']

frequency_online_per_pdv = order_detail_online.groupby('PointOfSaleId').size().reset_index(name='Frequency_online')
order_detail_sorted = order_detail_sorted.merge(frequency_online_per_pdv, on='PointOfSaleId', how='left')



# # Crear variable temporal
# start_date = order_detail_sorted['OrderDate'].min()
# end_date = order_detail_sorted['OrderDate'].max() + pd.Timedelta(days=1)
# date_range = pd.date_range(start=start_date, end=end_date)
# date_order_detail_sorted = pd.DataFrame(date_range, columns=['OrderDate'])
# date_order_detail_sorted['DayCount'] = (date_order_detail_sorted['OrderDate'] - start_date).dt.days + 1
# order_detail_sorted = pd.merge_asof(order_detail_sorted.sort_values('OrderDate'), date_order_detail_sorted, on='OrderDate', direction='forward')

# Crear variable de tipo de cupón
order_detail_sorted['CouponDiscountAmt'] = order_detail_sorted['CouponDiscountAmt'].astype(float)
order_detail_sorted['CouponDiscountPct'] = pd.to_numeric(order_detail_sorted['CouponDiscountPct'], errors='coerce')

def clasificar_descuento(row):
    if pd.notna(row['CouponDiscountAmt']) and pd.notna(row['CouponDiscountPct']) and pd.notna(row['NormalizedCoupon']):
        if row['CouponDiscountAmt'] != 0 and row['CouponDiscountPct'] == 0:
            return 'Fijo'
        elif row['CouponDiscountPct'] != 0 and row['NormalizedCoupon'] != 0:
            return 'Porcentual'
    return 'NoCupon'

order_detail_sorted['Coupon_type'] = order_detail_sorted.apply(clasificar_descuento, axis=1)


### GUARDAMOS EL PARQUET SIN VARIABLES NUEVAS
# save_path = r'C:\Users\ctrujils\order_detail_sorted_limpio.parquet'
# order_detail_sorted.to_parquet(save_path, index=False)

# Recurrencia a la compra de cupones
# print(order_detail_sorted[order_detail_sorted["Origin"] == "OFFLINE"]['PointOfSaleId'])

order_detail_sorted=order_detail_sorted[order_detail_sorted['Sellout']>0]

order_detail_sorted['CouponCode'] = order_detail_sorted['CouponCode'].replace(['', 'None', '0', '0.0', '0.00', 'NaN', 0], np.nan)


sales_coupon = order_detail_sorted[order_detail_sorted['NormalizedCoupon'] != 'No cupón'].groupby('PointOfSaleId')['Sellout'].sum()
sales_online = order_detail_sorted[order_detail_sorted['Origin'] == 'Online'].groupby('PointOfSaleId')['Sellout'].sum()
pctg_coupon_used = (sales_coupon / sales_online) * 100
pctg_coupon_used_order_detail_sorted = pctg_coupon_used.reset_index(name='PctgCouponUsed')


order_detail_sorted = order_detail_sorted.merge(pctg_coupon_used_order_detail_sorted, on='PointOfSaleId', how='left')
order_detail_sorted['PctgCouponUsed'] = order_detail_sorted['PctgCouponUsed'].fillna(0)

duplicados = pctg_coupon_used_order_detail_sorted['PointOfSaleId'].duplicated().any()
print(f"Hay duplicados: {duplicados}")



#Dias transcurridos desde la ultima compra online (acumulado)

order_detail_online = order_detail_online.sort_values(by=['PointOfSaleId', 'NameDistributor', 'OrderDate','Code'])

order_detail_online['DaysBetweenPurchases'] = order_detail_online.groupby(['PointOfSaleId', 'NameDistributor'])['OrderDate'].diff().dt.days

order_detail_online['DaysBetweenPurchases'] = order_detail_online['DaysBetweenPurchases'].fillna(0)

order_detail_online['CumulativeAvgDaysBetweenPurchases'] = order_detail_online.groupby(['PointOfSaleId', 'NameDistributor'])['DaysBetweenPurchases'].expanding().mean().reset_index(level=[0, 1], drop=True)
order_detail_online_subset = order_detail_online[['PointOfSaleId', 'OrderDate', 'NameDistributor', 'Code', 'DaysBetweenPurchases', 'CumulativeAvgDaysBetweenPurchases']]

num_filas_antes = order_detail_sorted.shape[0]


order_detail_sorted = pd.merge(order_detail_sorted, 
                               order_detail_online_subset, 
                               on=['PointOfSaleId', 'OrderDate', 'NameDistributor', 'Code'],
                               how='left')


num_filas_despues = order_detail_sorted.shape[0]
assert num_filas_antes == num_filas_despues, f"El número de filas cambió después del merge: antes={num_filas_antes}, después={num_filas_despues}"


# Booleana de ultima compra online

order_detail_sorted = order_detail_sorted.sort_values(by=['PointOfSaleId', 'NameDistributor', 'OrderDate'])
order_detail_sorted['IsOnline'] = order_detail_sorted['Origin'] == 'Online'
order_detail_sorted['LastPurchaseOnline'] = order_detail_sorted.groupby(['PointOfSaleId', 'NameDistributor'])['IsOnline'].shift(1)
order_detail_sorted['LastPurchaseOnline'] = order_detail_sorted['LastPurchaseOnline'].fillna(False).astype(bool)



distribuidores_por_pdv = order_detail_sorted.groupby('PointOfSaleId')['NameDistributor'].nunique().reset_index()
multiples_distribuidores = distribuidores_por_pdv[distribuidores_por_pdv['NameDistributor'] > 1]
order_detail_sorted = order_detail_sorted[~order_detail_sorted['PointOfSaleId'].isin(multiples_distribuidores['PointOfSaleId'])]

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
    g["Digitalizacion"] = g.IsOnline.rolling(5, min_periods=1).sum()
    g["ForwardOnline"] = g.IsOnline \
                           .iloc[::-1].rolling(5, min_periods=0).sum().iloc[::-1] \
                           .apply(lambda x: x > 0)
    digitalizacion.append(g)

order_detail_sorted = pd.concat(digitalizacion)

#Eliminamos pedidos vacíos
order_detail_sorted = order_detail_sorted[order_detail_sorted['Code'].str.strip() != '']



print('ya')
# # # # Guardar archivo final
# save_path = r'C:\Users\ctrujils\order_detail_sorted_v2.parquet'
# order_detail_sorted.to_parquet(save_path, index=False)



save_path = r'C:\Users\ctrujils\order_detail_sorted_normalizado.parquet'
order_detail_sorted.to_parquet(save_path, index=False)

print('fin de verdad')