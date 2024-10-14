import pandas as pd
# from sklearn.linear_model import LinearRegression, BayesianRidge, ARDRegression, SGDClassifier
# from sklearn.decomposition import PCA
# from sklearn.naive_bayes import GaussianNB
import numpy as np
import matplotlib.pyplot as plt
import itertools as it
import pickle

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


def generate_data_for_simulation(num_lags, coupon_level, poly_grade):
    out_list = [np.eye(num_lags) * (coupon_level ** k)
                for k in range(1, poly_grade + 1)]
    return np.concatenate(out_list, axis=1)


def make_plot(model_name, results_por_nivel, labels, y_label):
    plot_name = model_name.replace(" ", "_").lower()
    plt.figure().set_figwidth(10)
    plt.boxplot(results_por_nivel, labels=labels)
    plt.title(f'Simulaciones: {model_name}')
    plt.xlabel('Euros')
    plt.ylabel(y_label)
    plt.grid(True)
    plt.savefig(f"plots/pdvs/{plot_name}.jpg")
    plt.close()


def get_indexes(feat_cols, col_name):
    start = list(it.dropwhile(lambda x: not x.startswith(col_name), feat_cols))
    start_index = len(feat_cols) - len(start)
    end = list(it.dropwhile(lambda x: x.startswith(col_name), start))
    end_index = len(feat_cols) - len(end)
    return start_index, end_index


def make_simulations(model_name, model, levels, y_label="Retornos"):
    coef_cupones = model.coef_

    sigma_cupones = np.diag(model.sigma_)

    data = [generate_data_for_simulation(6, cl, 2)
            for cl in range(1, levels, 2)]

    results_por_nivel = []
    # Iterar sobre las matrices generadas
    for matrix in data:
        resultados_nivel = []
        for _ in range(1000):
            c2 = np.random.normal(coef_cupones,
                                  np.sqrt(sigma_cupones))
            r = np.dot(matrix, c2).sum() - matrix[0, 0]
            resultados_nivel.append(r)
        results_por_nivel.append(resultados_nivel)

    labels = [i for i in [1] + [k for k in range(2, levels, 2)]]
    make_plot(model_name, results_por_nivel, labels, y_label)


def compute_returns(df, pdv, model):
    start_index, end_index = get_indexes(feat_cols, "CouponDiscountAmt")
    df = df.copy()[df.PointOfSaleId == pdv]
    df.year = df.OrderDate.apply(lambda x: x.year)
    investment = df[df.year == 2023]['CouponDiscountAmt'].sum()
    returns = \
        np.dot(df[feat_cols].iloc[:, start_index:end_index][(df.year == 2023)],
               model.coef_,  #[start_index:end_index],
               ).sum()
    print(investment)
    print(returns)

    returns = []
    for k in range(600):
        c = np.random.normal(
            model.coef_,  # [start_index:end_index],
            np.diag(model.sigma_))  # [start_index:end_index])
        r = np.dot(
            df[feat_cols].iloc[:, start_index:end_index][(df.year == 2023)],
            c).sum()
        returns.append((r - investment) / investment)
    return returns


def plot_returns(returns, model_name):
    plot_name = model_name.replace(" ", "_").lower()
    plt.hist(returns, bins=30, density=True,
             color='pink',
             edgecolor='red')
    plt.title("Retorno de la inversión en cupones")
    plt.xlabel(f"Retorno porcentual: {plot_name}")
    plt.ylabel("Frequencia")
    plt.savefig(f"plots/pdvs/returns_{plot_name}.jpg")
    plt.close()
    

with open("submodelos_pdv_2.pkl", "rb") as f:
    modelos_pdv = pickle.load(f)

with open("modelo_general.pkl", "rb") as f:
    modelo_general = pickle.load(f)

df_modelo = pd.read_parquet("data/df_modelo.parquet")
df_modelo.year = df_modelo.OrderDate.apply(lambda x: x.year)

df_pca = pd.read_parquet("data/pca_coefs_2.parquet")
df_pca = df_pca[df_pca.NumOrders >= 5]
quantiles_pca = df_pca.PC1.quantile([1, 0.75, 0.5, 0.25, 0],
                                    interpolation="nearest")

for qk in quantiles_pca.to_dict().keys():
    quantile = quantiles_pca[qk]
    print(quantile)
    point_of_sale_id = \
        df_pca[df_pca.PC1 == quantile].PointOfSaleId.tolist()[-1]
    print(point_of_sale_id)
    model = [x for x in modelos_pdv
             if x['pdv']['PointOfSaleId'] == point_of_sale_id][0]
    print(model)
    model_name = f"Quantile {qk} {point_of_sale_id}"
    # make_simulations(model_name, model["modelo"], 16)
    returns = compute_returns(df_modelo, point_of_sale_id, model["modelo"])
    plot_returns(returns, model_name)
