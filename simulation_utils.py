import itertools as it

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


def get_indexes(feat_cols, col_name):
    start = list(it.dropwhile(lambda x: not x.startswith(col_name), feat_cols))
    start_index = len(feat_cols) - len(start)
    end = list(it.dropwhile(lambda x: x.startswith(col_name), start))
    end_index = len(feat_cols) - len(end)
    return start_index, end_index
