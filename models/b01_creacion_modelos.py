import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import os, sys

sys.path.append(os.getcwd()) # Esto es para agregar al path la ruta de ejecución actual y poder importar respecto a la ruta del proyecto, desde donde se debe ejecutar el código
from functions.evaluar_modelos import evaluate_model
# Leer datos ---------------------------------------- 

features_train = pd.read_csv("./files/data/intermediate/a02_features_train.csv")
target_train = pd.read_csv("./files/data/intermediate/a02_target_train.csv")
features_test = pd.read_csv("./files/data/intermediate/a02_features_test.csv")
target_test = pd.read_csv("./files/data/intermediate/a02_target_test.csv")
target_train = target_train.values.ravel()
target_test = target_test.values.ravel()

# Tuning models ---------------------------------------- 

print("Random Forest Classifier")

rfc = RandomForestClassifier(
    criterion='entropy', 
    max_depth=8,             
    random_state=12345       
)
rfc.fit(features_train, target_train)

evaluate_model(rfc, features_train, target_train, features_test, target_test,"modeling_output/figures/roc_curve_randomforestclassifier.png")


# Save model ---------------------------------------- 


joblib.dump(
        rfc,
        f"modeling_output/model_fit/b01_model_rf.joblib"
        )

