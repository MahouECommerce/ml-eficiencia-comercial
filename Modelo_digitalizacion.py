import pandas as pd
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler
import numpy as np
import itertools as it
from sklearn.linear_model import BayesianRidge, ARDRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
import numpy as np
import matplotlib.pyplot as plt
# Cargar los datos
path = r'C:\Users\ctrujils\order_detail_sorted.parquet'
order_detail_sorted = pd.read_parquet(path)

from sklearn.preprocessing import PolynomialFeatures



order_detail_sorted['OrderDate'] = pd.to_datetime(order_detail_sorted['OrderDate'])
order_detail_sorted=order_detail_sorted.sort_values(by='OrderDate',ascending=True)
## Creación de variables lagueadas
def create_lagged_variables(df, lag_variable, lag_periods, groupby_cols=None):
    """
    Función para crear variables laguadas en un DataFrame.
    
    Parámetros:
    - df: DataFrame original.
    - lag_variable: Nombre de la columna para la cual crear las variables laguadas.
    - lag_periods: Lista de períodos de desplazamiento.
    - groupby_cols: Lista de columnas para agrupar (opcional).
    
    Retorna:
    - DataFrame con las variables laguadas agregadas.
    """
    if groupby_cols:
        for lag in lag_periods:
            df[f'{lag_variable}_LAG_{lag}'] = df.groupby(groupby_cols)[lag_variable].shift(lag)
    else:
        for lag in lag_periods:
            df[f'{lag_variable}_LAG_{lag}'] = df[lag_variable].shift(lag)
    
    return df

import pandas as pd
order_detail_sorted = create_lagged_variables(order_detail_sorted, 'CouponDiscountAmt', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], ['Name'])
order_detail_sorted = create_lagged_variables(order_detail_sorted, 'Sellout', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], ['NameDistributor', 'Name'])


# Generate the list of column names dynamically
coupon_columns = [f'CouponDiscountAmt_LAG_{i}' for i in range(1, 11)]
sellout_columns = [f'Sellout_LAG_{i}' for i in range(1, 11)]

# amount_columns = [f'Amount_LAG_{i}' for i in range(1, 11)]

all_columns = coupon_columns + sellout_columns

order_detail_sorted[all_columns] = order_detail_sorted[all_columns].fillna(0)
order_detail_sorted[all_columns] = order_detail_sorted[all_columns].replace('',0)
order_detail_sorted[all_columns] = order_detail_sorted[all_columns].apply(pd.to_numeric, errors='coerce')

# Transformar la variable 'DayCount' a sus términos polinómicos
poly = PolynomialFeatures(degree=2, include_bias=False)
daycount_poly = poly.fit_transform(order_detail_sorted[['DayCount']])
poly_feature_names = poly.get_feature_names_out(['DayCount'])
daycount_poly_df = pd.DataFrame(daycount_poly, columns=poly_feature_names)
order_detail_sorted = pd.concat([order_detail_sorted, daycount_poly_df], axis=1)
poly = PolynomialFeatures(degree=2, include_bias=False)



# Transformar la variable 'coupondiscountamt' a sus términos polinómicos
order_detail_sorted['CouponDiscountAmt'].replace('', 0, inplace=True)
order_detail_sorted['CouponDiscountAmt'].fillna(0, inplace=True)
order_detail_sorted['CouponDiscountAmt'].astype(float)
coupondiscountamt_poly = poly.fit_transform(order_detail_sorted[['CouponDiscountAmt']])
poly_feature_names = poly.get_feature_names_out(['CouponDiscountAmt'])
coupondiscountamt_poly_df = pd.DataFrame(coupondiscountamt_poly, columns=poly_feature_names)
order_detail_sorted = order_detail_sorted.drop(columns=['CouponDiscountAmt'])

order_detail_sorted = pd.concat([order_detail_sorted, coupondiscountamt_poly_df], axis=1)



def create_polynomial_variables(df, poly_variables, degree=2):
    """
    Función para crear variables polinómicas en un DataFrame.

    Parámetros:
    - df: DataFrame original.
    - poly_variables: Lista de nombres de columnas para las cuales crear las variables polinómicas.
    - degree: Grado del polinomio (por defecto 2 para cuadráticas).

    Retorna:
    - DataFrame con las variables polinómicas agregadas.
    """
    for var in poly_variables:
        for d in range(2, degree + 1):
            df[f'{var}_POLY_{d}'] = df[var] ** d
    
    return df


# variables polinómicas de grado 2 y grado 3 
order_detail_sorted = create_polynomial_variables(order_detail_sorted, all_columns, degree=2)
df=order_detail_sorted.copy()
df=pd.get_dummies(df, columns=['tipologia'], prefix='Tip')
df=pd.get_dummies(df, columns=['Origin'], prefix='Orig')
df=pd.get_dummies(df, columns=['Coupon_type'], prefix='Type')

features= df[['Frequency_online','PctgCouponUsed',
       'CumulativeAvgDaysBetweenPurchases', 'ForwardOnline', 
       'DayCount', 'DayCount^2', 'CouponDiscountAmt', 'CouponDiscountAmt^2',
       'CouponDiscountAmt_LAG_1_POLY_2', 'CouponDiscountAmt_LAG_2_POLY_2',
       'CouponDiscountAmt_LAG_3_POLY_2', 'CouponDiscountAmt_LAG_4_POLY_2',
       'CouponDiscountAmt_LAG_5_POLY_2', 'Digitalizacion',
       'Tip_Bar Tradicional', 'Tip_Cervecería',
       'Tip_Discoteca', 'Tip_Establecimiento de Tapeo', 'Tip_No Segmentado',
       'Tip_Noche Temprana', 'Tip_Pastelería/Cafetería/Panadería',
       'Tip_Restaurante', 'Tip_Restaurante de Imagen',
       'Tip_Restaurante de Imagen con Tapeo', 'Tip_Tap room', 'Orig_OFFLINE','Type_Porcentual']]


# # Matriz de correlacion de nuestras variables
# correlation_matrix = features.corr()

# # Mostrar la matriz de correlación
# print(correlation_matrix)

# # Si deseas visualizar la matriz de correlación usando un mapa de calor
# import seaborn as sns
# import matplotlib.pyplot as plt

# plt.figure(figsize=(12, 10))
# sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
# plt.title('Matriz de Correlación de Variables')
# plt.show()


# Selección de la variable objetivo y características
target_digitalizacion = df['Digitalizacion']
features.drop(columns=['Digitalizacion'],inplace=True)


# Rellenar valores nulos en DayCount
features['DayCount'] = features['DayCount'].fillna(0)

# Indicar el rango de índices de interés de las características relacionadas con "CouponDiscountAmt"
# Función para encontrar los índices de columnas relacionadas con cupones
def get_indexes(feat_cols, col_name):
    start = list(it.dropwhile(lambda x: not x.startswith(col_name), feat_cols))
    start_index = len(feat_cols) - len(start)
    end = list(it.dropwhile(lambda x: x.startswith(col_name), start))
    end_index = len(feat_cols) - len(end)
    return start_index, end_index

start_index, end_index = get_indexes(features.columns, "CouponDiscountAmt")

# Ajuste del modelo Bayesian Ridge
modelo_digitalizacion = BayesianRidge()
modelo_digitalizacion.fit(features.iloc[:, start_index:end_index], target_digitalizacion)

# Simulaciones basadas en coeficientes del modelo
def generate_data_for_simulation(num_lags, coupon_level, poly_grade):
    out_list = [np.eye(num_lags) * (coupon_level ** k) for k in range(1, poly_grade + 1)]
    return np.concatenate(out_list, axis=1)

def make_plot(model_name, results_por_nivel, labels, y_label):
    plt.figure().set_figwidth(10)
    plt.boxplot(results_por_nivel, labels=labels)
    plt.title(f'Simulaciones: {model_name}')
    plt.xlabel('Euros')
    plt.ylabel(y_label)
    plt.grid(True)
    plt.show()

def make_simulations(model_name, model, levels, y_label="Digitalización"):
    coef_cupones = modelo_digitalizacion.coef_[start_index:end_index]
    
    # Verificar el tamaño de coef_cupones
    print(f"Tamaño de coef_cupones: {coef_cupones.shape}")
    
    sigma_cupones = np.diag(model.sigma_) if hasattr(model, 'sigma_') else np.zeros_like(coef_cupones)
    
    # Verificar el tamaño de sigma_cupones
    print(f"Tamaño de sigma_cupones: {sigma_cupones.shape}")

    data = [generate_data_for_simulation(6, cl, 2) for cl in range(1, levels, 2)]

    results_por_nivel = []
    
    # Verificar las dimensiones de las matrices generadas
    for matrix in data:
        print(f"Tamaño de matrix: {matrix.shape}")
        resultados_nivel = []
        
        for _ in range(1000):
            c2 = np.random.normal(coef_cupones, np.sqrt(sigma_cupones)) if sigma_cupones.size > 0 else coef_cupones
            
            # Verificar el tamaño de c2
            print(f"Tamaño de c2: {c2.shape}")
            
            # Multiplicación y acumulación
            r = np.dot(matrix, c2).sum() - matrix[0, 0]
            resultados_nivel.append(r)
        
        results_por_nivel.append(resultados_nivel)

    labels = [i for i in [1] + [k for k in range(2, levels, 2)]]
    make_plot(model_name, results_por_nivel, labels, y_label)

# Ejecutar simulaciones para el modelo de digitalización
make_simulations("Modelo Digitalización", modelo_digitalizacion, 10, "Digitalización")


print('fin')