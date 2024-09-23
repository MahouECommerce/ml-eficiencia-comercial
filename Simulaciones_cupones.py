import pickle
import pandas as pd
# Cargar la lista de diccionarios desde el archivo pickle
with open('submodelos_pdv.pkl', 'rb') as file:
    modelos_pdv = pickle.load(file)

# df=pd.read_csv('filtered.csv', sep=';'

pdvs = df['PointOfSaleId'].unique()
# Definir una función para predecir ventas sin cupones
def prediccion_base(pdv, df_pdv):
    modelo = modelos_pdv[pdv]
    
    # Modificar el DataFrame con valores de cupón simulados (cero cupones)
    df_pdv_sin_cupones = df_pdv.copy()
    df_pdv_sin_cupones['CouponDiscountAmt'] = 0
    df_pdv_sin_cupones['CouponDiscountPct'] = 0
    
    # Predecir
    X = df_pdv_sin_cupones[['CouponDiscountAmt', 'CouponDiscountPct', 'Frequency_online', 'PctgCouponUsed', 'Sellout', 'Digitalizacion']]
    y_pred = modelo.predict(X)
    
    return y_pred.sum()  # Total de predicciones "online"

# Probar para un PDV
for pdv in pdvs[:5]:  # Ejemplo para los primeros 5 PDVs
    ventas_sin_cupones = prediccion_base(pdv, df[df['PointOfSaleId'] == pdv])
    print(f"PDV: {pdv}, Ventas sin cupones: {ventas_sin_cupones}")
















print('fin')