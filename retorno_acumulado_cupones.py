import pandas as pd
import pickle
import warnings
import numpy as np
from simulation_utils import get_indexes, feat_cols
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
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
    # print(discount_mean)
    # print(investment)
    returns = \
        np.dot(df[feat_cols].iloc[:, start_index:end_index],
               model.coef_[start_index:end_index],
               ).sum()
    # print(investment)
    # print(returns)

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

corte_cupon = [99, 97, 95, 93, 91, 89, 87, 85, 83, 81, 79, 77, 75, 73, 71]


grupo_nombres = []
cupon_medio_list = []
cuantiles_list = []
diferencia_cuantiles_list = []

pdvs_acumulados=set()

for corte in corte_cupon:
    df_corte_actual=df_pdv_coupon_usage[df_pdv_coupon_usage['PctgCouponUsed']>=corte]
    pdvs_acumulados.update(df_corte_actual['PointOfSaleId'])

    df_grupo_acumulado=df[df['PointOfSaleId'].isin(pdvs_acumulados)]
    df_grupo_rest=df[~df['PointOfSaleId'].isin(pdvs_acumulados)]


    if df_grupo_acumulado.empty:
        print(f"El DataFrame del grupo acumulado está vacío para el corte {corte}%")
        continue
    if df_grupo_rest.empty:
        print(f"El DataFrame del grupo restante está vacío para el corte {corte}%")
        continue


    # Entrenamos el modelo para los acumulados

  
    br_acumulados = BayesianRidge(fit_intercept=False)
    br_acumulados.fit(df_grupo_acumulado[feat_cols], df_grupo_acumulado['Sellout'])

    #Calcular y almacenar cupon medio 
    cupon_medio = df_grupo_acumulado['CouponDiscountAmt'].mean()
    cupon_medio_list.append(cupon_medio)
    grupo_nombres.append(f"Acumulados ({corte}% cupones)")
    
    # Calcular y almacenar los retornos
    returns_acumulados = compute_returns(df_grupo_acumulado, br_acumulados)
    cuantiles_acumulados = np.quantile(np.array(returns_acumulados), [0.05, 0.5, 0.95])
    diferencia_cuantiles_list.append(cuantiles_acumulados[1] - cuantiles_acumulados[0])

    # Entrenar el modelo para el resto 
    
    br_resto = BayesianRidge(fit_intercept=False)
    br_resto.fit(df_grupo_rest[feat_cols], df_grupo_rest['Sellout'])
    
    returns_resto = compute_returns(df_grupo_rest, br_resto)
    cuantiles_resto = np.quantile(np.array(returns_resto), [0.05, 0.5, 0.95])

    diferencia_cuantiles = cuantiles_acumulados - cuantiles_resto
    diferencia_cuantiles_list.append(diferencia_cuantiles)

with open('resultados_acumulados.pkl', "wb") as file:
        resultados = {
            'grupo_nombres': grupo_nombres,
            'cupon_medio_list': cupon_medio_list,
            'cuantiles_list': cuantiles_list,
            'diferencia_cuantiles_list': diferencia_cuantiles_list
        }
        pickle.dump(resultados, file)
print("Resultados guardados en el archivo pickle.")  

# with open('resultados_acumulados.pkl', "rb") as file:
#     resultados = pickle.load(file)
#     grupo_nombres = resultados['grupo_nombres']
#     cupon_medio_list = resultados['cupon_medio_list']
#     cuantiles_list = resultados['cuantiles_list']
#     diferencia_cuantiles_list = resultados['diferencia_cuantiles_list']
#     print("Resultados cargados del archivo pickle.")

diferencia_cuantiles_arr = np.array([np.array(dif) for dif in diferencia_cuantiles_list if isinstance(dif, np.ndarray)])

# Extraer la mediana de las diferencias directamente (índice 1 del array de cuantiles)
mediana_diferencias = diferencia_cuantiles_arr[:, 1]

scaler = MinMaxScaler(feature_range=(0, 1))

# Ajustar el escalador usando todo el conjunto de diferencias centradas
scaler.fit(mediana_diferencias[:,np.newaxis])

diferencia_cuantiles_normalized = np.zeros_like(diferencia_cuantiles_arr)

# Bucle para transformar cada columna de `diferencia_cuantiles_arr`
for i in range(diferencia_cuantiles_arr.shape[1]):
    diferencia_cuantiles_normalized[:, i] = scaler.transform(diferencia_cuantiles_arr[:, i].reshape(-1, 1)).flatten()



resultados_df = pd.DataFrame({
    'Grupo': grupo_nombres,
    'Cupon_Medio': cupon_medio_list,
    'Dif_Cuantiles_5%': [dif[0] if isinstance(dif, (list, np.ndarray)) else dif for dif in diferencia_cuantiles_normalized],
    'Dif_Cuantiles_50%': [dif[1] if isinstance(dif, (list, np.ndarray)) else dif for dif in diferencia_cuantiles_normalized],
    'Dif_Cuantiles_95%': [dif[2] if isinstance(dif, (list, np.ndarray)) else dif for dif in diferencia_cuantiles_normalized]})


print('fin')

resultados_df['Acumulado_Dif_Cuantiles_5%'] = resultados_df['Dif_Cuantiles_5%'].cumsum()
resultados_df['Acumulado_Dif_Cuantiles_50%'] = resultados_df['Dif_Cuantiles_50%'].cumsum()
resultados_df['Acumulado_Dif_Cuantiles_95%'] = resultados_df['Dif_Cuantiles_95%'].cumsum()

import matplotlib.pyplot as plt
resultados_df['Grupo'] = [f'Punto de corte {corte}%' for corte in corte_cupon ]
plt.figure(figsize=(10, 6))

# Graficar los cuantiles del 5%, 50% (mediana) y 95% para cada grupo
# plt.plot(resultados_df['Grupo'], resultados_df['Acumulado_Dif_Cuantiles_5%'], label='Cuantil 5%', color='#ff9896', linewidth=2)
plt.plot(resultados_df['Grupo'], resultados_df['Acumulado_Dif_Cuantiles_50%'], label='Mediana (50%)', color='#d62728', linewidth=2)
# plt.plot(resultados_df['Grupo'], resultados_df['Acumulado_Dif_Cuantiles_95%'], label='Cuantil 95%', color='#7f7f7f', linewidth=2)

# Título y etiquetas de los ejes
plt.title('Diferencias Normalizadas entre Cuantiles por Grupo', fontsize=16)
plt.xlabel('Grupo', fontsize=14)
plt.ylabel('Diferencia Normalizada', fontsize=14)

# Rotar las etiquetas del eje x para que sean más legibles
plt.xticks(rotation=45, ha='right')

# Agregar una leyenda para entender qué representa cada línea
plt.legend(fontsize=12)

# Mostrar una cuadrícula para hacer el gráfico más fácil de leer
plt.grid(True, linestyle='--', alpha=0.7)

# Ajustar el gráfico para que todos los elementos se vean bien
plt.tight_layout()

# Mostrar el gráfico
# plt.show()
print('fin')




import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np


# # Crear una lista con los nombres de los grupos que queremos analizar
# grupos_interes = ['Acumulados (95% cupones)', 'Acumulados (80% cupones)', 'Acumulados (75% cupones)']


# scaler = MinMaxScaler(feature_range=(0, 1))

# # Iterar sobre cada punto de corte y graficar la distribución para returns acumulados y resto
# for corte in grupos_interes:
#     # Filtrar los datos de los acumulados y resto
#     df_grupo_acumulado = df[df['PointOfSaleId'].isin(resultados[f'{corte}'])]
#     df_grupo_rest = df[~df['PointOfSaleId'].isin(resultados[f'acumulados_{corte}'])]
    
#     # Escalar los returns acumulados y los returns del resto
#     returns_acumulados_scaled = scaler.fit_transform(df_grupo_acumulado['returns'].values.reshape(0, 1)).flatten()
#     returns_resto_scaled = scaler.fit_transform(df_grupo_rest['returns'].values.reshape(0, 1)).flatten()

#     # Crear una figura para cada punto de corte
#     plt.figure(figsize=(10, 6))

#     # Graficar los histogramas para los datos escalados
#     plt.hist(returns_acumulados_scaled, bins=30, alpha=0.6, label='Returns Acumulados', color='#ff7f0e', edgecolor='black')
#     plt.hist(returns_resto_scaled, bins=30, alpha=0.6, label='Returns Resto', color='#1f77b4', edgecolor='black')

#     # Añadir título y etiquetas a cada gráfica
#     plt.title(f'Distribución de Returns para Corte {corte}%', fontsize=16)
#     plt.xlabel('Returns Normalizados', fontsize=14)
#     plt.ylabel('Frecuencia', fontsize=14)

#     # Añadir una leyenda para identificar las distribuciones
#     plt.legend(fontsize=12)

#     # Mostrar la cuadrícula para mejor lectura del gráfico
#     plt.grid(True, linestyle='--', alpha=0.7)

#     # Ajustar el gráfico para evitar sobreposiciones
#     plt.tight_layout()

#     # Mostrar el gráfico
#     plt.show()