import pandas as pd
import pickle
import warnings
import numpy as np
from simulation_utils import get_indexes, feat_cols
from sklearn.linear_model import BayesianRidge
warnings.filterwarnings('ignore')

# feat_cols = [c for c in feat_cols if c != "Orig_OFFLINE"]

# correction = 40000


def compute_returns(df, model):
    start_index, end_index = get_indexes(feat_cols, "CouponDiscountAmt")
    df.year = df.OrderDate.apply(lambda x: x.year)
    investment = df[df.year == 2023]['CouponDiscountAmt'].sum()
    # investment = df['CouponDiscountAmt'].sum()
    # investment = 575810.43
    discount_mean = df['CouponDiscountAmt'].mean()
    print(discount_mean)
    print(investment)
    returns = \
        np.dot(df[feat_cols].iloc[:, start_index:end_index],
               model.coef_[start_index:end_index],
               ).sum()
    print(investment)
    print(returns)

    returns = []
    for k in range(600):
        c = np.random.normal(
            model.coef_[start_index:end_index],
            np.diag(model.sigma_)[start_index:end_index])
        r = np.dot(
            df[feat_cols].iloc[:, start_index:end_index],
            c).sum()
        returns.append((r - investment) / investment)
    return returns


# dormidos = pd.read_csv(
#     "data/11102024_clientes_dormidos.csv", sep=";").PointOfSaleId
# offline = pd.read_csv(
#     "data/11102024_clientes_solo_offline.csv", sep=";").PointOfSaleId
# cuponeros = pd.read_csv(
#     "data/14102024_clientes_cuponeros_v2.csv", sep=";").PointOfSaleId
# optimos = pd.read_csv(
#     "data/14102024_muestra_optima.csv", sep=",").PointOfSaleId

path = r'C:\Users\ctrujils\eficiencia_promocional\0. Datos\\'

# Definir los nombres de los archivos
file_dormidos = '11102024_clientes_dormidos.csv'
file_offline = '11102024_clientes_solo_offline.csv'
file_cuponeros = '14102024_clientes_cuponeros_v2.csv'
file_optimos = '14102024_muestra_optima.csv'
file_modelo= 'df_modelo.parquet'
pickle_modelo='modelo_general.pkl'


# Leer los archivos utilizando la misma estructura
dormidos = pd.read_csv(f'{path}{file_dormidos}', sep=';').PointOfSaleId
offline = pd.read_csv(f'{path}{file_offline}', sep=';').PointOfSaleId
cuponeros = pd.read_csv(f'{path}{file_cuponeros}', sep=';').PointOfSaleId
optimos = pd.read_csv(f'{path}{file_optimos}', sep=',').PointOfSaleId



df = pd.read_parquet(f'{path}{file_modelo}')
df.CouponDiscountAmt.sum()

modelo_path=f'{path}{pickle_modelo}'

with open(modelo_path, "rb") as f:
    modelo_general = pickle.load(f)

#### DataFrame por pdv y % de cuponero

df_pdv_coupon_usage = df.groupby('PointOfSaleId')['PctgCouponUsed'].mean().reset_index()


#Definimos umbral 
corte_cupon=[100, 99, 98, 95, 93, 90, 88, 85, 82, 80]

#Creamos diccionario para almacenar dfs
df_por_corte={}
enumerate_corte=enumerate(corte_cupon, start=1)
for idx , coupon in enumerate_corte:
   df_corte= df_pdv_coupon_usage[df_pdv_coupon_usage['PctgCouponUsed']> coupon]

   df_por_corte[f'df_cuponeros_corte_{coupon}']=df_corte


df_filtrados={}
## Sacamos los dfs con los pdvs de cada df del dccionario de df_por_corte
for i, df_pdv in df_por_corte.items():
    df_corte=df[df['PointOfSaleId'].isin(df_pdv['PointOfSaleId'])]

    df_filtrados[f'{i}']=df_corte

#  guardar en csv
# for key, filtered_df in df_filtrados.items():
#     filtered_df.to_csv(f'{key}.csv', index=False)

# grupos = [
#     {"nombre": "dormidos", "df": dormidos},
#     # {"nombre": "offline", "df": offline},
#     {"nombre": "cuponeros", "df": df_cuponeros.PointOfSaleId},
#     {"nombre": "optimos", "df": optimos},]




grupos = [{"nombre": key, "df": df_filtrados[key]['PointOfSaleId']} for key in df_filtrados]


# for g in grupos:
#     print(g["nombre"])
#     gdf = df[df.PointOfSaleId.isin(g["df"].tolist())]
#     br = BayesianRidge(fit_intercept=False)
#     br.fit(gdf[feat_cols], gdf.Sellout)
#     print(gdf.shape)
    
#     # Calcular y mostrar los retornos
#     returns = compute_returns(gdf, br)
#     print(f"Cuantiles de los retornos para el grupo {g['nombre']}: {np.quantile(np.array(returns), [0.05, 0.5, 0.95])}")



import matplotlib.pyplot as plt

# Crear listas para almacenar los resultados
grupo_nombres = []
cupon_medio_list = []
cuantiles_list = []

for g in grupos:
    
    gdf_ids = g["df"].tolist() if not g["df"].empty else []
    
    gdf = df[df['PointOfSaleId'].isin(gdf_ids)]
    
    
    if gdf.shape[0] == 0:
        continue

    
    br = BayesianRidge(fit_intercept=False)
    br.fit(gdf[feat_cols], gdf['Sellout'])
    # print(f"Tamaño del grupo '{g['nombre']}': {gdf.shape}")
    
    # Calcular y almacenar el cupón medio
    cupon_medio = gdf['CouponDiscountAmt'].mean()
    cupon_medio_list.append(cupon_medio)
    grupo_nombres.append(g['nombre'])
    
    # Calcular y mostrar los retornos
    returns = compute_returns(gdf, br)
    cuantiles = np.quantile(np.array(returns), [0.05, 0.5, 0.95])
    cuantiles_list.append(cuantiles)
    print(f"Cuantiles de los retornos para el grupo {g['nombre']}: {cuantiles}")

# Crear DataFrame para visualizar los resultados
import pandas as pd
resultados_df = pd.DataFrame({
    'Grupo': grupo_nombres,
    'Cupon_Medio': cupon_medio_list,
    'Cuantiles_5%': [cu[0] for cu in cuantiles_list],
    'Cuantiles_50%': [cu[1] for cu in cuantiles_list],
    'Cuantiles_95%': [cu[2] for cu in cuantiles_list]
})


# Graficar el cupón medio por grupo
plt.figure(figsize=(10, 5))
plt.bar(resultados_df['Grupo'], resultados_df['Cupon_Medio'], color='skyblue')
plt.xlabel('Grupo')
plt.ylabel('Cupón Medio')
plt.title('Cupón Medio por Grupo')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.show()

# Graficar los cuantiles por grupo
plt.figure(figsize=(12, 6))
plt.plot(resultados_df['Grupo'], resultados_df['Cuantiles_5%'], label='5% Cuantil', marker='o', color='red')
plt.plot(resultados_df['Grupo'], resultados_df['Cuantiles_50%'], label='50% Cuantil (Mediana)', marker='o', color='green')
plt.plot(resultados_df['Grupo'], resultados_df['Cuantiles_95%'], label='95% Cuantil', marker='o', color='blue')
plt.xlabel('Grupo')
plt.ylabel('Retornos')
plt.title('Distribución de los Cuantiles de los Retornos por Grupo')
plt.legend()
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.show()


print('fin')