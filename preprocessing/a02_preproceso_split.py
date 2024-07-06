# Librerias ---------------------------------------- 

import pandas as pd
import os, sys
sys.path.append(os.getcwd()) # Esto es para agregar al path la ruta de ejecución actual y poder importar respecto a la ruta del proyecto, desde donde se debe ejecutar el código

# Loading data ---------------------------------------- 

df_final_telecom_cleaned = pd.read_csv("./files/data/intermediate/a01_df_final_telecom_cleaned.csv")

# Splitting data into sets ---------------------------------------- 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import KNNImputer

features = df_final_telecom_cleaned.drop('cancel', axis=1) #x
target = df_final_telecom_cleaned['cancel'] #y
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.20, random_state=12345)

# Ordinal_Encoder ------------------------------------------------------------------
columns_object=df_final_telecom_cleaned.select_dtypes(include='object').columns
enc = OrdinalEncoder()
X_train[columns_object] = enc.fit_transform(X_train[columns_object])
X_test[columns_object] = enc.transform(X_test[columns_object])

#Imputación knn ----------------------------------------------------------------------
imputer_kk=KNNImputer(n_neighbors=5)
#Creamos el data frame con los valores imputados nan
imputed_X_train=pd.DataFrame(imputer_kk.fit_transform(X_train))
imputed_X_valid=pd.DataFrame(imputer_kk.transform(X_test))
#Obetnemos el nombre de las columnas 
imputed_X_train.columns=X_train.columns
imputed_X_valid.columns=X_test.columns
columns_to_int = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                     'TechSupport', 'StreamingTV', 'StreamingMovies', 'MultipleLines']
#El método round(0) redondea al entero más cercano.
imputed_X_train[columns_to_int] = imputed_X_train[columns_to_int].round(0).astype('Int64')
imputed_X_valid[columns_to_int]= imputed_X_valid[columns_to_int].round(0).astype('Int64')
X_train=imputed_X_train.copy()
X_test=imputed_X_valid.copy()

X_test.info()


# Save data ---------------------------------------- 

X_train.to_csv("./files/data/intermediate/a02_features_train.csv", index=False)
y_train.to_csv("./files/data/intermediate/a02_target_train.csv", index=False)
X_test.to_csv("./files/data/intermediate/a02_features_test.csv", index=False)
y_test.to_csv("./files/data/intermediate/a02_target_test.csv", index=False)
