# MAGIC En este notebook desarrolaremos un modelo de regresión (Bayessian Ridge)
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import asyncio
import mlflow.pyfunc
import itertools as it
import numpy as np

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
len(order_detail_sorted[order_detail_sorted.year >= 2023].PointOfSaleId.unique())

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

descriptive = pd.DataFrame(descriptive)
descriptive.shape
descriptive[
    descriptive.had_offline &
    (descriptive.before_online_ratio < descriptive.after_online_ratio)
]
descriptive[
    descriptive.had_offline &
    (3 < descriptive.after_digitalizacion) &
    (descriptive.before_digitalizacion < descriptive.after_digitalizacion)    
]

descriptive[
    descriptive.had_offline &
    (3 < descriptive.after_digitalizacion) &
    (descriptive.before_digitalizacion < descriptive.after_digitalizacion)    
].after_digitalizacion.mean()

descriptive[
    descriptive.had_offline &
    (3 < descriptive.after_digitalizacion) &
    (descriptive.before_digitalizacion < descriptive.after_digitalizacion)    
].before_digitalizacion.mean()

order_detail_sorted.Digitalizacion.mean()


pdv_means = order_detail_sorted[["PointOfSaleId", "Digitalizacion", "Sellout", "Online", "CouponDiscountAmt"]] \
    .groupby("PointOfSaleId") \
    .agg(
        {"Digitalizacion": "mean", "Sellout": "mean", "Online": "sum", "CouponDiscountAmt": "mean"}) \
    .reset_index()
pdv_means[["Digitalizacion", "Sellout", "Online", "CouponDiscountAmt"]].corr()
pdv_means[pdv_means.CouponDiscountAmt > 50][["Digitalizacion", "Sellout", "Online", "CouponDiscountAmt"]].corr()
pdv_means[pdv_means.CouponDiscountAmt < 10][["Digitalizacion", "Sellout", "Online", "CouponDiscountAmt"]].corr()
pdv_means[pdv_means.Sellout > 400][["Digitalizacion", "Sellout", "Online", "CouponDiscountAmt"]].corr()



pdv_means[pdv_means.CouponDiscountAmt > 70].Digitalizacion.mean()
pdv_means.Digitalizacion.mean()

pdv_means[pdv_means.CouponDiscountAmt > 50].Sellout.mean()
pdv_means.Sellout.mean()

pdv_means[pdv_means.CouponDiscountAmt > 50].Online.sum() / pdv_means[pdv_means.CouponDiscountAmt > 50].shape[0]
pdv_means.Online.sum() / pdv_means.shape[0]


pdv_means[pdv_means.Digitalizacion > 6].CouponDiscountAmt.mean()
pdv_means.CouponDiscountAmt.mean()

pdv_means[pdv_means.Digitalizacion > 6].Sellout.mean()
pdv_means.Sellout.mean()
