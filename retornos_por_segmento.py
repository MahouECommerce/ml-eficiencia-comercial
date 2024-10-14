import pandas as pd
import pickle
import numpy as np
from simulation_utils import get_indexes, feat_cols

correction = 40000


def compute_returns(df, model):
    start_index, end_index = get_indexes(feat_cols, "CouponDiscountAmt")
    df.year = df.OrderDate.apply(lambda x: x.year)
    # investment = df[df.year == 2023]['CouponDiscountAmt'].sum()
    investment = df['CouponDiscountAmt'].sum()
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


dormidos = pd.read_csv(
    "data/11102024_clientes_dormidos.csv", sep=";").PointOfSaleId
offline = pd.read_csv(
    "data/11102024_clientes_solo_offline.csv", sep=";").PointOfSaleId
cuponeros = pd.read_csv(
    "data/14102024_clientes_cuponeros_v2.csv", sep=";").PointOfSaleId
optimos = pd.read_csv(
    "data/14102024_muestra_optima.csv", sep=",").PointOfSaleId

grupos = [
    {"nombre": "dormidos", "df": dormidos},
    # {"nombre": "offline", "df": offline},
    {"nombre": "cuponeros", "df": cuponeros},
    {"nombre": "optimos", "df": optimos},
]

df = pd.read_parquet("data/df_modelo.parquet")
df.CouponDiscountAmt.sum()
with open("modelo_general.pkl", "rb") as f:
    modelo_general = pickle.load(f)

for g in grupos:
    print(g["nombre"])
    gdf = df[df.PointOfSaleId.isin(g["df"].tolist())]
    print(gdf.shape)
    returns = compute_returns(gdf, modelo_general)
    print(np.quantile(np.array(returns), [0.05, 0.5, 0.95]))
