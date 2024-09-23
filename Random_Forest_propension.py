# Importar librerías
import pandas as pd
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score

# Leer datos
df_path = r'C:\Users\ctrujils\order_detail_sorted.parquet'
df = pd.read_parquet(df_path)

# Tratamiento de columnas categoricas, y valores nulos

df.fillna(0, inplace=True)  
df=pd.get_dummies(df, columns=['tipologia'], prefix='Tip')
df=pd.get_dummies(df, columns=['Origin'], prefix='Orig')
df=pd.get_dummies(df, columns=['Coupon_type'], prefix='Type')




# Seleción de columnas para el modelo 

feat_cols=df[[
'DayCount',
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
'Sellout',
'Frequency_online',
'PctgCouponUsed',
'CumulativeAvgDaysBetweenPurchases',
'LastPurchaseOnline',
'Digitalizacion']]


# División de X/Y
X = feat_cols
y = df['ForwardOnline']

#Normalización de variables
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


rf = RandomForestClassifier(class_weight='balanced', random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

# print("Classification Report:\n", classification_report(y_test, y_pred))
# print("AUC-ROC:", roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1]))


#Feature importance

# importances = rf.feature_importances_
# print(importances)



# Validación cruzada

cv_scores = cross_val_score(rf, X_scaled, y, cv=5, scoring='f1')
# print("Cross-validated F1 scores:", cv_scores)
# print("Mean F1 score:", cv_scores.mean())


# CURVA DE VALIDAZION

from sklearn.model_selection import validation_curve

# Función para plotear curvas de validación
def plot_validation_curve(estimator, X, y, param_name, param_range, title="Curvas de Validación", cv=None, n_jobs=-1):
    train_scores, test_scores = validation_curve(
        estimator, X, y, param_name=param_name, param_range=param_range, cv=cv, n_jobs=n_jobs
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(param_range, train_scores_mean, 'o-', color='r', label='Precisión en entrenamiento')
    plt.plot(param_range, test_scores_mean, 'o-', color='g', label='Precisión en validación')
    plt.fill_between(param_range, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color='r')
    plt.fill_between(param_range, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color='g')
    plt.title(title)
    plt.xlabel(param_name)
    plt.ylabel('Precisión')
    plt.legend(loc='best')
    plt.grid()
    plt.show()

# Ejemplo para max_depth
param_range = np.arange(1, 21)  
plot_validation_curve(rf, X_train, y_train, param_name='max_depth', param_range=param_range, cv=5)





print('fin')