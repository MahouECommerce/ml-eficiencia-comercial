# MAGIC En este notebook desarrolaremos un modelo de regresión (Bayessian Ridge)
import sys
import pandas as pd
from sklearn.linear_model import BayesianRidge, ARDRegression, LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io_utils as iou
from returns import pipeline, unsafe
import asyncio
import mlflow.pyfunc
import itertools as it
import numpy as np

# config = iou.read_config()
# container_name = pipeline.flow(
#     config.map(lambda x: x["container_name"]).unwrap(),
#     unsafe.unsafe_perform_io)
# client = config.bind(lambda c: iou.build_blob_service_client()(c))

# order_detail_sorted_path="cleaned_data/order_detail_model.parquet"
# future_order_detail_sorted= client.bind(
#     iou.read_parquet_into_df(container_name, order_detail_sorted_path))
# order_detail_sorted = pipeline.flow(
#     future_order_detail_sorted.awaitable(),
#     asyncio.run,
#     lambda x: x.unwrap(),
#     unsafe.unsafe_perform_io)
# order_detail_sorted.to_parquet("data/order_detail_model.parquet")
# order_detail_sorted.columns

def create_lagged_variables(df, lag_variable, lag_periods, groupby_cols=None):
    if groupby_cols:
        for lag in lag_periods:
            df[f'{lag_variable}_LAG_{lag}'] = \
                df.groupby(groupby_cols)[lag_variable].shift(lag).fillna(0)
    else:
        for lag in lag_periods:
            df[f'{lag_variable}_LAG_{lag}'] = df[lag_variable].shift(lag).fillna(0)
    
    return df

order_detail_sorted = pd.read_parquet("data/order_detail_model.parquet")
order_detail_sorted["Online"] = order_detail_sorted.Origin.apply(lambda x: x == "Online")
gt = order_detail_sorted.groupby(['NameDistributor', 'PointOfSaleId'])
digitalizacion = []
for _, g in gt:
    g["Digitalizacion"] = g.Online.rolling(10, min_periods=1).sum()
    digitalizacion.append(g)
order_detail_sorted = pd.concat(digitalizacion)
order_detail_sorted = create_lagged_variables(
    order_detail_sorted, 'CouponDiscountPct',
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], ['NameDistributor', 'Name'])
order_detail_sorted.to_csv("data/digitalizacion.csv", sep=";")

def get_index(df, starts_with):
    return df[[c for c in df.columns if c.startswith(starts_with)]] \
        .apply(lambda x: x.sum() > 0, axis=1)
order_detail_sorted[["Coupon_type", "CouponDescription"]].groupby("Coupon_type").count()
coupon_index = get_index(order_detail_sorted, 'CouponDiscountAmt')
pdvs_with_coupon = order_detail_sorted[coupon_index].PointOfSaleId.unique()
len(pdvs_with_coupon)
len(order_detail_sorted.PointOfSaleId.unique())

order_detail_sorted.columns

descriptive = []
for pdv in pdvs_with_coupon:
    data = order_detail_sorted[coupon_index & (order_detail_sorted.PointOfSaleId == pdv)]
    first_coupon = data.OrderDate.min()
    before = order_detail_sorted[(order_detail_sorted.OrderDate < first_coupon) &
                                 (order_detail_sorted.PointOfSaleId == pdv)]
    after = order_detail_sorted[(order_detail_sorted.OrderDate >= first_coupon) &
                                (order_detail_sorted.PointOfSaleId == pdv)]
    if before.shape[0] and after.shape[0]:
        results = {"pdv": pdv,
                   "before_digitalizacion": before.Digitalizacion.mean(),
                   "before_pedidos_online": before.Online.sum(),
                   "before_pedidos": before.Online.count(),
                   "before_online_ratio":
                   before.Online.sum() if before.Online.sum() == 0 else before.Online.sum() / before.Online.count(),
                   "before_sellout": before.Sellout.mean(),
                   "before_coupon": before.CouponDiscountAmt.mean(),                   
                   "after_digitalizacion": after.Digitalizacion.mean(),
                   "after_pedidos_online": after.Online.sum(),
                   "after_pedidos": after.Online.count(),
                   "after_online_ratio":
                   after.Online.sum() if after.Online.sum() == 0 else after.Online.sum() / after.Online.count(),
                   "after_sellout": after.Sellout.mean(),
                   "after_coupon": after.CouponDiscountAmt.mean(),
                   "had_offline": before.Online.sum() < before.Online.count()}
        descriptive.append(results)
        if (before.Online.sum() < before.Online.count()) and (
                before.Digitalizacion.mean() < after.Digitalizacion.mean()):
            print(results)
            print("-"*100)

descriptive = pd.DataFrame(descriptive)
descriptive[descriptive.had_offline]
order_detail_sorted[
    order_detail_sorted.PointOfSaleId=="CLI0042747CLI0050757"][["Digitalizacion", "Online", "CouponDiscountAmt"]]

pct_index = get_index(order_detail_sorted, 'CouponDiscountPct')
fixed_index = pd.concat([coupon_index, pct_index], axis=1) \
                .apply(lambda x: x[0] and not x[1], axis=1)
pct_index.sum()
fixed_index.sum()
no_coupon_index = coupon_index.apply(lambda x: not x)
fixed_model_index = pd.concat([no_coupon_index, fixed_index], axis=1) \
                      .apply(lambda x: x[0] or x[1], axis=1)
pct_model_index = pd.concat([no_coupon_index, pct_index], axis=1) \
                      .apply(lambda x: x[0] or x[1], axis=1)

no_coupon_index.sum()
fixed_index.sum()
pct_index.sum()
order_detail_sorted.shape
order_detail_sorted.Coupon_type.unique()
order_detail_sorted[["Coupon_type", "CouponDiscountAmt"]].groupby("Coupon_type").mean()
order_detail_sorted[["Coupon_type", "CouponDiscountAmt"]].groupby("Coupon_type").count()

order_detail_sorted.columns
df=order_detail_sorted.copy()
# MAGIC ####Label encoding de "tipologia"
df=pd.get_dummies(df, columns=['tipologia'], prefix='Tip')
df=pd.get_dummies(df, columns=['Origin'], prefix='Orig')
df=pd.get_dummies(df, columns=['Coupon_type'], prefix='Type')
df=pd.get_dummies(df, columns=['season'], prefix='season')

# MAGIC ### Filtro Sellout < 3000
df['Sellout'].describe()

coupon_cols = [c for c in order_detail_sorted.columns if c.startswith("CouponDiscountAmt")]
coupon_poly_cols = ['CouponDiscountAmt^2'] + list(it.dropwhile(lambda x: not x.endswith("POLY_2"), coupon_cols))
coupon_cols = list(set(coupon_cols) - set(coupon_poly_cols))

order_detail_sorted[order_detail_sorted.CouponDiscountAmt > 0].CouponDiscountAmt.describe()
order_detail_sorted[order_detail_sorted.CouponDiscountAmt > 100].CouponDiscountAmt.describe()
coupon_cutoff = 96
def filter_lags(df, coupon_cutoff):
    for col in coupon_cols:
        df[col] = np.where(df[col] <= coupon_cutoff, df[col], 0)
    for col in coupon_poly_cols:
        df[col] = np.where(df[col] <= (coupon_cutoff ** 2), df[col], 0)
    return df

df = df[df["CouponDiscountAmt"] <= coupon_cutoff]
df = filter_lags(df, coupon_cutoff)
df.to_csv("filtered.csv", sep=";")

# df=df[df['year']==2023]
# df=df[df['Sellout']]

# prueba=features[features.columns[17:39]]
# prueba
# coef_with_weight

# MAGIC %md
# MAGIC #### Comparación sellout inversión
# df['Sellout'].sum()
# np.dot(prueba,coef_no_weight[17:39]).sum()
# df['CouponDiscountAmt'].sum()

# MAGIC ####Selección de columnas para el modelo
feat_cols = [
'season_Otoño',
'season_Primavera',
'season_Verano',
'DayCount',
'DayCount^2',
'Tip_Bar Tradicional',
'Tip_Cervecería',
'Tip_Discoteca',
'Tip_Establecimiento de Tapeo',
'Tip_No Segmentado',
'Tip_Noche Temprana',
'Tip_Pastelería/Cafetería/Panadería',
'Tip_Restaurante',
'Tip_Restaurante de Imagen',
'Tip_Restaurante de Imagen con Tapeo',
'Orig_OFFLINE',
'Type_Porcentual',
'CouponDiscountAmt',
'CouponDiscountAmt_LAG_1',
'CouponDiscountAmt_LAG_2',
'CouponDiscountAmt_LAG_3',
'CouponDiscountAmt_LAG_4',
'CouponDiscountAmt_LAG_5',
'CouponDiscountAmt_LAG_6',
'CouponDiscountAmt_LAG_7',
'CouponDiscountAmt_LAG_8',
'CouponDiscountAmt_LAG_9',
'CouponDiscountAmt_LAG_10',
'CouponDiscountAmt^2',
'CouponDiscountAmt_LAG_1_POLY_2',
'CouponDiscountAmt_LAG_2_POLY_2',
'CouponDiscountAmt_LAG_3_POLY_2',
'CouponDiscountAmt_LAG_4_POLY_2',
'CouponDiscountAmt_LAG_5_POLY_2',
'CouponDiscountAmt_LAG_6_POLY_2',
'CouponDiscountAmt_LAG_7_POLY_2',
'CouponDiscountAmt_LAG_8_POLY_2',
'CouponDiscountAmt_LAG_9_POLY_2',
'CouponDiscountAmt_LAG_10_POLY_2',
'Sellout_LAG_1',
'Sellout_LAG_2',
'Sellout_LAG_3',
'Sellout_LAG_4',
'Sellout_LAG_5',
'Sellout_LAG_6',
'Sellout_LAG_7',
'Sellout_LAG_8',
'Sellout_LAG_9',
'Sellout_LAG_10',
'Sellout_LAG_1_POLY_2',
'Sellout_LAG_2_POLY_2', 
'Sellout_LAG_3_POLY_2', 
'Sellout_LAG_4_POLY_2',
'Sellout_LAG_5_POLY_2', 
'Sellout_LAG_6_POLY_2', 
'Sellout_LAG_7_POLY_2',
'Sellout_LAG_8_POLY_2', 
'Sellout_LAG_9_POLY_2', 
'Sellout_LAG_10_POLY_2',
]

target_col = 'Sellout'
features = df[feat_cols]
target = df[target_col]
target_online = df.Orig_OFFLINE.apply(lambda x: not x)
target_digitalizacion = df.Digitalizacion

def imprimir_nombres_caracteristicas(feat_cols):
    for indice, caracteristica in enumerate(feat_cols):
        print(f"Índice {indice}: {caracteristica}")

def get_indexes(feat_cols, col_name):
    start = list(it.dropwhile(lambda x: not x.startswith(col_name), feat_cols))
    start_index = len(feat_cols) - len(start)
    end = list(it.dropwhile(lambda x: x.startswith(col_name), start))
    end_index = len(feat_cols) - len(end)
    return start_index, end_index

start_index, end_index = get_indexes(feat_cols, "CouponDiscountAmt")
# Llamada a la función con la lista feat_cols

df['DayCount']=df['DayCount'].fillna(0)
na_counts = features.isna().sum()
na_columns = na_counts[na_counts > 0]
na_columns

# MAGIC ### Ajuste Sample_weight para desbalanceo de muestras
pedidos_con_cupon = order_detail_sorted[order_detail_sorted['CouponDescription'] != 'NoCupon']['Code'].value_counts()
pedidos_sin_cupon = order_detail_sorted[order_detail_sorted['CouponDescription'] == 'NoCupon']['Code'].value_counts()

# Calcular número de pedidos con y sin cupon
n_with_coupon = len(pedidos_con_cupon)
n_no_coupon = len(pedidos_sin_cupon)
weight_no_coupon = 1
weight_with_coupon = n_no_coupon / n_with_coupon


#Creación del array de pesos
feature_balance=features['CouponDiscountAmt']
sample_weight = np.where(feature_balance == 0, weight_no_coupon, weight_with_coupon)

# MAGIC ####Aplicación del modelo y entrenamiento

#FIT MODELO SIN PESOS
model_no_weight = BayesianRidge(fit_intercept=False)
model_no_weight.fit(features, target)
model_no_weight.coef_[start_index:end_index]
ardr = ARDRegression(fit_intercept=False)
ardr.fit(features, target)
ardr.coef_[start_index:end_index]
modelo_digitalizacion = BayesianRidge()
modelo_digitalizacion.fit(features.iloc[:, start_index:end_index], target_digitalizacion)
modelo_digitalizacion.coef_
np.diag(modelo_digitalizacion.sigma_)
np.diag(modelo_digitalizacion.sigma_)[start_index:end_index]
nb = GaussianNB()
nb.fit(features.iloc[:, start_index:end_index], target_online)
online_predict = nb.predict(features.iloc[:, start_index:end_index])
target_online
online_predict
df[target_online & np.vectorize(lambda x: not x)(online_predict)].CouponDiscountAmt
online_predict

model_no_weight.coef_[start_index:end_index]
ardr.coef_[start_index:end_index]
model_no_weight.coef_[start_index:end_index]

df['Sellout'].sum()
investment = df[df.year==2023]['CouponDiscountAmt'].sum()
print(investment)
np.dot(features.iloc[:, start_index:end_index][(df.year==2023)],
       model_no_weight.coef_[start_index:end_index],
       ).sum()

returns = []
for k in range(600):
    c = np.random.normal(
        model_no_weight.coef_[start_index:end_index],
        np.diag(model_no_weight.sigma_)[start_index:end_index])
    r = np.dot(features.iloc[:, start_index:end_index][(df.year==2023)], c).sum()
    returns.append((r - investment) / investment)

plt.hist(returns, bins=30)
plt.savefig("plots/returns.jpg")
plt.close()
model_no_weight.coef_[start_index:end_index]
coef_cupones_2 = ardr.coef_[start_index:(end_index - 11)]
plt.bar(range(len(coef_cupones_2)), coef_cupones_2, color='skyblue', edgecolor='black')
plt.title('Coeficientes del Modelo')
plt.xlabel('Índice del Coeficiente')
plt.ylabel('Valor del Coeficiente')
plt.xticks(range(len(coef_cupones_2)))
plt.savefig("plots/coeficiente.jpg")
plt.close()

coef_cupones_2 = model_no_weight.coef_[(start_index + 11):end_index]
plt.bar(range(len(coef_cupones_2)), coef_cupones_2, color='skyblue', edgecolor='black')
plt.title('Coeficientes cuadráticos del Modelo')
plt.xlabel('Índice del Coeficiente')
plt.ylabel('Valor del Coeficiente')
plt.xticks(range(len(coef_cupones_2)))
plt.savefig("plots/coeficiente_2.jpg")
plt.close()


    
np.dot(features.iloc[:, start_index:end_index][(df.year==2023)],
       model_no_weight.coef_[start_index:end_index]).sum()
np.dot(features.iloc[:, start_index:end_index][(df.year==2023) &
                                               (df.CouponDiscountAmt < 20)],
       model_no_weight.coef_[start_index:end_index]).sum()
np.dot(features.iloc[:, start_index:end_index][(df.year==2023) &
                                               (df.CouponDiscountAmt >= 20)],
       model_no_weight.coef_[start_index:end_index]).sum()

def fit_model(features, target, target_digitalizacion):
    model = BayesianRidge(fit_intercept=False)
    model.fit(features, target)
    ardr = ARDRegression(fit_intercept=False)
    ardr.fit(features, target)
    modelo_digitalizacion = BayesianRidge(fit_intercept=False)
    modelo_digitalizacion.fit(features, target_digitalizacion)    
    return model, ardr, modelo_digitalizacion

def generate_data_for_simulation(num_lags, coupon_level, poly_grade):
    out_list = [np.eye(num_lags) * (coupon_level ** k)
                for k in range(1, poly_grade + 1)]
    return np.concatenate(out_list, axis=1)

def make_probability_plot(model_name, model):
    data = [generate_data_for_simulation(11, cl, 2)
            for cl in range(1, 80, 2)]
    results = []
    for d in data:
        results.append(model.predict_log_proba(d)[:, 1].mean())
    labels = [i for i in [1] + [k for k in range(2, 80, 2)]]
    plt.plot(labels, results)
    plt.title(f'Log-probabilidad compra online: {model_name}')
    plt.xlabel('Euros')
    plt.ylabel('Log-probabilidad')
    plot_name = model_name.replace(" ", "_").lower()
    plt.savefig(f"plots/log_proba_{plot_name}.png")
    plt.close()

def make_plot(model_name, results_por_nivel, labels, y_label):
    plot_name = model_name.replace(" ", "_").lower()
    plt.figure().set_figwidth(10)    
    plt.boxplot(results_por_nivel, labels=labels)
    plt.title(f'Simulaciones: {model_name}')
    plt.xlabel('Euros')
    plt.ylabel(y_label)
    plt.grid(True)
    plt.savefig(f"plots/{plot_name}.jpg")
    plt.close()
    
def make_simulations(model_name, model, levels, y_label="Retornos", diffs=True, digi=False):
    start_index, end_index = get_indexes(feat_cols, "CouponDiscountAmt")
    if not digi:
        coef_cupones = model.coef_[start_index:end_index]
    else:
        coef_cupones = model.coef_
    
    sigma_cupones = np.diag(model_no_weight.sigma_)[start_index:end_index]

    data = [generate_data_for_simulation(11, cl, 2)
            for cl in range(1, levels, 2)]

    results_por_nivel = []
    # Iterar sobre las matrices generadas
    for matrix in data:
        resultados_nivel = []
        for _ in range(1000):
            c2 = np.random.normal(coef_cupones, np.sqrt(sigma_cupones))
            r = np.dot(matrix, c2).sum() - matrix[0,0]
            resultados_nivel.append(r)
        results_por_nivel.append(resultados_nivel)

    labels = [i for i in [1] + [k for k in range(2, levels, 2)]]
    make_plot(model_name, results_por_nivel, labels, y_label)

    if diffs:
        results_por_diff = []
        # Iterar sobre las matrices generadas
        for k in range(1, len(data)):
            resultados_diff = []
            for _ in range(1000):
                c2 = np.random.normal(coef_cupones, np.sqrt(sigma_cupones))
                r = (np.dot(data[k], c2) - np.dot(data[k - 1], c2)).sum()
                resultados_diff.append(r)
            results_por_diff.append(resultados_diff)

        labels = [i for i in [k for k in range(2, levels, 2)]]
        make_plot(f"{model_name} diffs", results_por_diff, labels, y_label)
    
    
make_simulations("Modelo digitalizacion General", modelo_digitalizacion, 2, "Digitalizacion", False, True)
make_simulations("Modelo General", model_no_weight, coupon_cutoff + 2)
make_probability_plot("Modelo probabilidad digitalizacion", nb)

indices = {"fixed": fixed_model_index, "pct": pct_model_index}
for k in indices:
    f = features[indices[k]]
    t = target[indices[k]]
    td = target_digitalizacion[indices[k]]
    br, ardr, md = fit_model(f, t, td)
    make_simulations(f"Modelo digitalizacion {k}", md, 2, "Digitalizacion", False)
    make_simulations(f"Modelo general {k}", br, coupon_cutoff + 2)
    

years = [2023, 2024]
tipologias = [c for c in features.columns if c.startswith("Tip")
              if c not in ["Tip_Tap room", 'Tip_Restaurante de Imagen con Tapeo']]

for colname in tipologias:
    f = features[features[colname]]
    t = target[features[colname]]
    to = target_online[features[colname]]
    br, ardr, nb = fit_model(f, t, to)
    try:
        make_simulations(f"Modelo {colname}", br, 22)
        make_probability_plot(f"Modelo General {colname}", nb)                    
    except Exception as e:
        print(e)
    for y in years:
        try:
            ft = features[features[colname] & (df.year == y)]
            tt = target[features[colname] & (df.year == y)]
            tto = target_online[features[colname] & (df.year == y)]
            br, ardr, nb = fit_model(ft, tt, tto)
            make_simulations(f"Modelo {colname} {y}", br, 22)
            make_probability_plot(f"Probabilidad Online {colname} {y}", nb)            
        except Exception as e:
            print(e)
            
