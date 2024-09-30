import pickle
import pandas as pd
import numpy as np
import itertools as it
# Cargar la lista de diccionarios desde el archivo pickle
with open('submodelos_pdv.pkl', 'rb') as file:
    modelos_pdv = pickle.load(file)

df=pd.read_csv('filtered.csv', sep=';')
from features_list import feat_cols

##### SIMULACIONES OPTIMIZACION CUPONES : pdv_cuponero = 'CLI0032489' 
pdv_cuponero = 'CLI0032489' 


## sellout medio del pdv historico 
def calcular_sellout_medio_pdv(df, pdv_id):
    # Filtrar el DataFrame por el PDV específico
    df_pdv = df[df['PointOfSaleId'] == pdv_id]
    
    # Calcular la media de la columna 'Sellout'
    sellout_medio = df_pdv['Sellout'].mean()
    
    return sellout_medio


sellout_medio_pdv = calcular_sellout_medio_pdv(df, pdv_cuponero)

print(f'Sellout medio del PDV {pdv_cuponero}: {sellout_medio_pdv}')


pdv = 'CLI0032489'
#### SIMULACION CUPON 

 # Buscar el modelo correspondiente al PDV
modelo = None
for modelo_info in modelos_pdv:
    if modelo_info['pdv'] == pdv:
        modelo = modelo_info['modelo']
        break
    # Verificar si se encontró el modelo
if modelo is None:
    print(f'No se encontró el modelo para el PDV {pdv}.')
    
def generate_data_for_simulation(num_lags, coupon_level, poly_grade):
    out_list = [np.eye(num_lags) * (coupon_level ** k)
                for k in range(1, poly_grade + 1)]
    return np.concatenate(out_list, axis=1)

def get_indexes(feat_cols, col_name):
    start = list(it.dropwhile(lambda x: not x.startswith(col_name), feat_cols))
    start_index = len(feat_cols) - len(start)
    end = list(it.dropwhile(lambda x: x.startswith(col_name), start))
    end_index = len(feat_cols) - len(end)
    return start_index, end_index

def simulacion_con_cupones(pdv, modelo, df, feat_cols, descuento_pct):
    # Filtrar el DataFrame para el PDV específico
    df_pdv = df[df['PointOfSaleId'] == pdv].copy()
    
    # Calcular la media de la columna 'Sellout'
    sellout_medio = df_pdv['Sellout'].mean()

    # Aplicar el descuento en la columna de descuento
    coupon_discount_amount = sellout_medio * descuento_pct / 100
    data=generate_data_for_simulation(6, coupon_discount_amount, 2)
    
    start_index, end_index = get_indexes(feat_cols, "CouponDiscountAmt")
    coef_cupones = modelo.coef_[start_index:end_index]
    sellout_euros=np.dot(data, coef_cupones).sum()
    print(sellout_euros)


simulacion_con_cupones(pdv, modelo, df, feat_cols, 20)




#     # Realizar predicciones
#     x_pdv = df_pdv[feat_cols]
#     y_pred_con_cupones = modelo.predict(x_pdv)

#     # Retornar el sellout promedio con el cupón aplicado
#     return y_pred_con_cupones.mean()

# # Simulación con un cupón del 20% para el PDV específico

# sellout_con_cupones = simulacion_con_cupones(pdv, modelos_pdv, df, feat_cols, 20)

# print(f'Sellout promedio con un cupón del 20% para el PDV {pdv}: {sellout_con_cupones}')




print('fin')


print('fin')

