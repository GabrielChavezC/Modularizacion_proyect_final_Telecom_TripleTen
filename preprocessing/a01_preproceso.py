import pandas as pd
import os, sys
sys.path.append(os.getcwd()) # Esto es para agregar al path la ruta de ejecución actual y poder importar respecto a la ruta del proyecto, desde donde se debe ejecutar el código

# Loading data ---------------------------------------- 

df_contract = pd.read_csv("./files/data/input/final_provider/contract.csv")
df_internet = pd.read_csv("./files/data/input/final_provider/internet.csv")
df_personal = pd.read_csv("./files/data/input/final_provider/personal.csv")
df_phone = pd.read_csv("./files/data/input/final_provider/phone.csv")


df_final_telecom_raw=df_contract.merge(df_personal,how='outer').merge(df_internet,how='outer').merge(df_phone,how='outer')

df_final_telecom_raw.info()
# Cleaning columns ---------------------------------------- 

def limpiar_columnas(dataset):
    dataset['cancel']=dataset['EndDate'].apply(lambda x: '0' if x == 'No' else '1').astype('int')
    dataset['TotalCharges'] = pd.to_numeric(dataset['TotalCharges'], errors='coerce').astype('float')
    #columnas que no proporcionan datos al modelado
    dataset.drop(columns=['customerID','BeginDate','EndDate'],inplace=True)
    return dataset

df_final_telecom = limpiar_columnas(df_final_telecom_raw)

# Eliminating duplicates ---------------------------------------- 

def drop_duplicates(dataset):
    dataset= dataset.drop_duplicates().reset_index(drop=True)   
    return dataset

df_final_telecom = drop_duplicates(df_final_telecom_raw)


# ...

# Checking NAs ---------------------------------------- 

# ...

# Guardar datos ---------------------------------------- 

df_final_telecom.to_csv("./files/data/intermediate/a01_df_final_telecom_cleaned.csv", index=False)

