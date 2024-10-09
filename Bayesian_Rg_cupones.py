# En este notebook desarrolaremos un modelo de regresión (Bayessian Ridge)
import pandas as pd
from sklearn.linear_model import BayesianRidge, ARDRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
import io_utils as iou
from returns import pipeline, unsafe
import asyncio
# import mlflow.pyfunc
import itertools as it
# import numpy as np

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


def create_day_count_df(order_detail_sorted):
    # Obtener la fecha de inicio y fin
    start_date = order_detail_sorted['OrderDate'].min()
    end_date = order_detail_sorted['OrderDate'].max()

    end_date_adjusted = end_date + pd.Timedelta(days=1)

    # Crear un rango de fechas que incluya el día después del final
    date_range = pd.date_range(start=start_date, end=end_date_adjusted)
    date_df = pd.DataFrame(date_range, columns=['OrderDate'])
    date_df['DayCount'] = (date_df['OrderDate'] - start_date).dt.days + 1

    return date_df


def create_lagged_variables(df, lag_variable, lag_periods, groupby_cols=None):
    if groupby_cols:
        for lag in lag_periods:
            df[f'{lag_variable}_LAG_{lag}'] = \
                df.groupby(groupby_cols)[lag_variable].shift(lag).fillna(0)
    else:
        for lag in lag_periods:
            df[f'{lag_variable}_LAG_{lag}'] = df[lag_variable] \
                .shift(lag).fillna(0)

    return df


def get_season(mes):
    if mes in [12, 1, 2]:
        return 'Invierno'
    elif mes in [3, 4, 5]:
        return 'Primavera'
    elif mes in [6, 7, 8]:
        return 'Verano'
    elif mes in [9, 10, 11]:
        return 'Otoño'
    else:
        return 'Desconocido'


order_detail_sorted = \
    pd.read_parquet("data/order_detail_sorted_normalizado.parquet")

order_detail_sorted = order_detail_sorted[
    ~order_detail_sorted.NameDistributor.isin(['Voldis Baleares', 'Ceres'])]
check = order_detail_sorted[["PointOfSaleId", "NameDistributor"]] \
    .groupby("PointOfSaleId") \
    .nunique()
order_detail_sorted = order_detail_sorted[
    ~order_detail_sorted.PointOfSaleId.isin(
        check[check.NameDistributor > 1].reset_index().PointOfSaleId)]

distribuidores_por_pdv = \
    order_detail_sorted.groupby('PointOfSaleId')['NameDistributor'].nunique().reset_index()
multiples_distribuidores = distribuidores_por_pdv[distribuidores_por_pdv['NameDistributor'] > 1]
multiples_distribuidores

order_detail_sorted = order_detail_sorted[order_detail_sorted.Sellout < 2000]
order_detail_sorted = order_detail_sorted[0 < order_detail_sorted.Sellout]

order_detail_sorted = \
    order_detail_sorted.sort_values(by="OrderDate", ascending=True)

order_detail_sorted['OrderDate'] = \
    pd.to_datetime(order_detail_sorted['OrderDate'])
order_detail_sorted['month'] = order_detail_sorted['OrderDate'].dt.month
order_detail_sorted['season'] = order_detail_sorted['month'].apply(get_season)

day_count_df = create_day_count_df(order_detail_sorted)
order_detail_sorted = order_detail_sorted.merge(
    day_count_df, how="left", on="OrderDate")

for c in ['DayCount', 'CouponDiscountAmt', 'Sellout']:
    new_name = f"{c}^2"
    order_detail_sorted[new_name] = order_detail_sorted[c]**2

order_detail_sorted["Online"] = \
    order_detail_sorted.Origin.apply(lambda x: x == "Online")

online_offline_pdv = order_detail_sorted[["PointOfSaleId", "Online"]] \
    .groupby("PointOfSaleId") \
    .sum() \
    .reset_index()
online_pdv = online_offline_pdv[online_offline_pdv.Online > 0]
order_detail_sorted = order_detail_sorted.merge(
    online_pdv["PointOfSaleId"], on="PointOfSaleId", how="right")

gt = order_detail_sorted.groupby(['NameDistributor', 'PointOfSaleId'])

digitalizacion = []
for _, g in gt:
    g["Digitalizacion"] = g.Online.rolling(5, min_periods=1).sum()
    g["ForwardOnline"] = g.Online.iloc[::-1] \
                                 .rolling(5, min_periods=0) \
                                 .sum().iloc[::-1] \
                                       .apply(lambda x: x > 0)
    digitalizacion.append(g)

order_detail_sorted = pd.concat(digitalizacion)
for c in ["CouponDiscountAmt", "Sellout",
          "CouponDiscountAmt^2", "Sellout^2"]:
    order_detail_sorted = create_lagged_variables(
        order_detail_sorted, c,
        [1, 2, 3, 4, 5], ['NameDistributor', 'Name'])


order_detail_sorted.to_csv("data/digitalizacion.csv", sep=";")
order_detail_sorted.columns


def get_index(df, starts_with):
    return df[[c for c in df.columns if c.startswith(starts_with)]] \
        .apply(lambda x: x.sum() > 0, axis=1)


order_detail_sorted[["Coupon_type", "CouponDescription"]] \
    .groupby("Coupon_type").count()
# coupon_index = get_index(order_detail_sorted, 'CouponDiscountAmt')
# pdvs_with_coupon = order_detail_sorted[coupon_index].PointOfSaleId.unique()

# pct_index = get_index(order_detail_sorted, 'CouponDiscountPct')
# fixed_index = pd.concat([coupon_index, pct_index], axis=1) \
#                 .apply(lambda x: x[0] and not x[1], axis=1)
# no_coupon_index = coupon_index.apply(lambda x: not x)
# fixed_model_index = pd.concat([no_coupon_index, fixed_index], axis=1) \
#                       .apply(lambda x: x[0] or x[1], axis=1)
# pct_model_index = pd.concat([no_coupon_index, pct_index], axis=1) \
#                       .apply(lambda x: x[0] or x[1], axis=1)

df = order_detail_sorted.copy()
df.columns
# MAGIC ####Label encoding de "tipologia"
df = pd.get_dummies(df, columns=['Origin'], prefix='Orig')
df = pd.get_dummies(df, columns=['NameDistributor'], prefix='Distributor')
df = pd.get_dummies(df, columns=['Coupon_type'], prefix='Type')
df = pd.get_dummies(df, columns=['season'], prefix='season')
df.columns
# MAGIC ### Filtro Sellout < 3000
df['Sellout'].describe()

coupon_cols = [c for c in order_detail_sorted.columns
               if c.startswith("CouponDiscountAmt")]
coupon_poly_cols = ['CouponDiscountAmt^2'] + list(it.dropwhile(
    lambda x: not x.endswith("POLY_2"), coupon_cols))
coupon_cols = list(set(coupon_cols) - set(coupon_poly_cols))

order_detail_sorted[order_detail_sorted.CouponDiscountAmt > 0] \
    .CouponDiscountAmt.describe()
order_detail_sorted[order_detail_sorted.CouponDiscountAmt > 100] \
    .CouponDiscountAmt.describe()


coupon_cutoff = 72


def filter_lags(df, coupon_cutoff):
    for col in coupon_cols:
        df[col] = np.where(df[col] <= coupon_cutoff, df[col], 0)
    for col in coupon_poly_cols:
        df[col] = np.where(df[col] <= (coupon_cutoff ** 2), df[col], 0)
    return df


df = df[df["CouponDiscountAmt"] <= coupon_cutoff]
df = filter_lags(df, coupon_cutoff)
df.to_csv("filtered.csv", sep=";")

feat_cols = [
    'season_Otoño',
    'season_Primavera',
    'season_Verano',
    'DayCount', 'DayCount^2',
    # 'Distributor_Bebicer',
    'Distributor_Voldis Coruña',
    'Distributor_Voldis Granada', 'Distributor_Voldis Madrid',
    'Distributor_Voldis Murcia', 'Distributor_Voldis Valencia',
    'Orig_OFFLINE',
    'Type_Porcentual',
    # 'DaysBetweenPurchases',
    # 'CumulativeAvgDaysBetweenPurchases',
    # "LastPurchaseOnline",
    # 'Digitalizacion',
    # 'ForwardOnline',
    'CouponDiscountAmt',
    'CouponDiscountAmt_LAG_1',
    'CouponDiscountAmt_LAG_2',
    'CouponDiscountAmt_LAG_3',
    'CouponDiscountAmt_LAG_4',
    'CouponDiscountAmt_LAG_5',
    'CouponDiscountAmt^2',
    'CouponDiscountAmt^2_LAG_1',
    'CouponDiscountAmt^2_LAG_2',
    'CouponDiscountAmt^2_LAG_3',
    'CouponDiscountAmt^2_LAG_4',
    'CouponDiscountAmt^2_LAG_5',
    'Sellout_LAG_1',
    'Sellout_LAG_2',
    'Sellout_LAG_3',
    'Sellout_LAG_4',
    'Sellout_LAG_5',
    'Sellout^2_LAG_1',
    'Sellout^2_LAG_2',
    'Sellout^2_LAG_3',
    'Sellout^2_LAG_4',
    'Sellout^2_LAG_5',
]
order_detail_sorted[["season", "Sellout"]].groupby("season").mean()
df.columns
target_col = 'Sellout'
features = df[feat_cols]
features = features.fillna(0)
target = df[target_col]
target_online = df.ForwardOnline
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

df['DayCount'] = df['DayCount'].fillna(0)
na_counts = features.isna().sum()
na_columns = na_counts[na_counts > 0]
na_columns

# MAGIC ### Ajuste Sample_weight para desbalanceo de muestras
# pedidos_con_cupon = \
#     order_detail_sorted[order_detail_sorted['NormalizedCoupon'] != 'No cupón']['Code'] \
#     .value_counts()
# pedidos_sin_cupon = \
#     order_detail_sorted[order_detail_sorted['NormalizedCoupon'] == 'No cupón']['Code'] \
#     .value_counts()

# Calcular número de pedidos con y sin cupon
# n_with_coupon = len(pedidos_con_cupon)
# n_no_coupon = len(pedidos_sin_cupon)
# weight_no_coupon = 1
# weight_with_coupon =  n_no_coupon / n_with_coupon

# #Creación del array de pesos
# feature_balance = features[coupon_cols + coupon_poly_cols].apply(
#     lambda x: 0 < x.sum(), axis= 1
# )
# feature_balance.sum()
# feature_balance.shape
# sample_weight = np.where(feature_balance,
#                          weight_with_coupon,                         
#                          weight_no_coupon,)
# sample_weight
# MAGIC ####Aplicación del modelo y entrenamiento

#FIT MODELO SIN PESOS
model_no_weight = BayesianRidge(fit_intercept=False)
model_no_weight.fit(features, target)
for z in zip(feat_cols, model_no_weight.coef_):
    print(z)
#model_no_weight.coef_[start_index:end_index]
model_no_weight.coef_[start_index:(start_index+6)].sum()

y_pred = model_no_weight.predict(features)
y_pred[y_pred < 0].shape
plt.scatter(y_pred, target)
plt.savefig("plots/errors.jpg")
plt.close()
target

ardr = ARDRegression(fit_intercept=False)
ardr.fit(features, target)
ardr.coef_[start_index:end_index]

df['Sellout'].sum()
df.year = df.OrderDate.apply(lambda x: x.year)
investment = df[df.year == 2023]['CouponDiscountAmt'].sum()
df.CouponDiscountAmt.describe()
print(investment)
np.dot(features.iloc[:, start_index:end_index][(df.year == 2023)],
       model_no_weight.coef_[start_index:end_index],
       ).sum()
df.CouponDiscountAmt[df.CouponDiscountAmt>0].describe()
df.CouponDiscountAmt[df.CouponDiscountAmt>70].describe()
features.iloc[:, start_index:end_index]
returns = []
for k in range(600):
    c = np.random.normal(
        model_no_weight.coef_[start_index:end_index],
        np.diag(model_no_weight.sigma_)[start_index:end_index])
    r = np.dot(features.iloc[:, start_index:end_index][(df.year==2023)], c).sum()
    returns.append((r - investment) / investment)

plt.hist(returns, bins=30, density=True,
         color='pink',
         edgecolor='red')
plt.title("Retorno de la inversión en cupones")
plt.xlabel("Retorno porcentual")
plt.ylabel("Frequencia")
plt.savefig("plots/returns_total.jpg")
plt.close()

coef_cupones_2 = model_no_weight.coef_[start_index:(start_index + 6)]
coef_cupones_2


def plot_coefs(coefs, model_name):
    plt.bar(range(len(coefs)), coefs,
            color='pink',
            edgecolor='red')

    plt.title(f'Coeficientes del Modelo {model_name}')
    plt.xlabel('Índice del Coeficiente')
    plt.ylabel('Valor del Coeficiente')
    plt.xticks(range(len(coef_cupones_2)))
    plt.savefig(f"plots/coeficiente_{model_name}.jpg")
    plt.close()


plot_coefs(coef_cupones_2, "modelo_general")

coef_cupones_2 = model_no_weight.coef_[(start_index + 11):end_index]
plt.bar(range(len(coef_cupones_2)), coef_cupones_2, color='skyblue', edgecolor='black')
plt.title('Coeficientes cuadráticos del Modelo')
plt.xlabel('Índice del Coeficiente')
plt.ylabel('Valor del Coeficiente')
plt.xticks(range(len(coef_cupones_2)))
plt.savefig("plots/coeficiente_2.jpg")
plt.close()


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
    data = [generate_data_for_simulation(6, cl, 2)
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

    data = [generate_data_for_simulation(6, cl, 2)
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
    
    
make_simulations("Modelo digitalizacion General", modelo_digitalizacion, 2,
                 "Digitalizacion", False, True)
make_simulations("Modelo General", model_no_weight, 60)
make_probability_plot("Modelo probabilidad digitalizacion", nb)

indices = {"fixed": fixed_model_index, "pct": pct_model_index}
for k in indices:
    f = features[indices[k]]
    t = target[indices[k]]
    td = target_digitalizacion[indices[k]]
    br, ardr, md = fit_model(f, t, td)
    print(br.coef_[start_index:(start_index+6)].sum())
    make_simulations(f"Modelo digitalizacion {k}", md, 2,
                     "Digitalizacion", False)
    make_simulations(f"Modelo general {k}", br, 76)
    plot_coefs(br.coef_[start_index:(start_index+6)], k)

years = [2023, 2024]
tipologias = [c for c in features.columns if c.startswith("Tip")
              if c not in ["Tip_Tap room", 'Tip_Restaurante de Imagen con Tapeo']]

# for colname in tipologias:
#     f = features[features[colname]]
#     t = target[features[colname]]
#     to = target_online[features[colname]]
#     br, ardr, nb = fit_model(f, t, to)
#     try:
#         make_simulations(f"Modelo {colname}", br, 22)
#         make_probability_plot(f"Modelo General {colname}", nb)                    
#     except Exception as e:
#         print(e)
#     for y in years:
#         try:
#             ft = features[features[colname] & (df.year == y)]
#             tt = target[features[colname] & (df.year == y)]
#             tto = target_online[features[colname] & (df.year == y)]
#             br, ardr, nb = fit_model(ft, tt, tto)
#             make_simulations(f"Modelo {colname} {y}", br, 22)
#             make_probability_plot(f"Probabilidad Online {colname} {y}", nb)            
#         except Exception as e:
#             print(e)
            
