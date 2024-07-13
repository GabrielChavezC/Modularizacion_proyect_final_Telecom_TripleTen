import streamlit as st
import pandas as pd 
import joblib
import os, sys
sys.path.append(os.getcwd()) 

#Configuracion
st.set_page_config(layout='wide')
st.title("Telecom")

st.sidebar.title('Prediccion')



col1, col2,col3 = st.columns(3)

with col1:
    type = st.radio(
    "Tiempo de contrato",
    ["Month-to-month", "One year", "Two year"],
    index=None,

)

    gender = st.radio(
    "Genero",
    ["femenino", "masculino"],
    index=None,

)
    
    partner = st.radio(
    "Pareja",
    ["Si", "No"],
    index=None,
)
    
    onlinebackup  = st.radio(
    "Almacenamiento Online",
    ["Si", "No"],
    index=None,
)
    
    streamingtv = st.radio(
    "Servicio de TV",
     ["Si", "No"],
    index=None,
)

with col2:
    paperlessbilling = st.radio(
    "Facturación Electrónica",
     ["Si", "No"],
    index=None,
)
    seniorcitizen = st.radio(
    "Adulto Mayor",
    ["Si", "No"],
    index=None,
)

    dependents = st.radio(
    "Dependientes",
    ["Si", "No"],
    index=None,
)
    deviceprotection = st.radio(
    "Proteccion de Dispositivos",
     ["Si", "No"],
    index=None,
)
    
    streamingmovies = st.radio(
    "Servicio de Películas",
     ["Si", "No"],
    index=None,
)
   
   
with col3:
    paymentmethod = st.radio(
    "Método de Pago",
    ["Bank Transfer", "Credit Card","Mailed check","Electronick check"],
    index=None,
)
    
    internetservice = st.radio(
    "Servicio de Internet",
    ["DSL", "Fibra Optica"],
    index=None,
)
    
    onlinesecurity  = st.radio(
    "Seguridad Online",
    ["Si", "No"],
    index=None,
)

    techsupport = st.radio(
    "Servicio Técnico",
     ["Si", "No"],
    index=None,
)
    multiplelines = st.radio(
    "Multiples Lineas",
     ["Si", "No"],
    index=None,
)
 

col1mcharges, col2tcharges = st.columns(2)

with col1mcharges:
     monthlycharges=st.text_input("Cargos Mensuales","Ingresa cargos mensuales")
     
with col2tcharges:
     totalcharges=st.text_input("Cargos Totales","Ingresa cargos Totales")
     


if st.sidebar.button('Prediccion'):
    # Convertir las entradas a un formato adecuado para el modelo
        data = {
            'Type': type,
            'PaperlessBilling': paperlessbilling,
            'PaymentMethod': paymentmethod,
            'MonthlyCharges': monthlycharges,
            'TotalCharges': totalcharges,
            'gender': gender,
            'SeniorCitizen': seniorcitizen,
            'Partner': partner,
            'Dependents': dependents,
            'InternetService': internetservice,
            'OnlineSecurity': onlinesecurity,
            'OnlineBackup': onlinebackup,
            'DeviceProtection': deviceprotection,
            'TechSupport': techsupport,
            'StreamingTV': streamingtv,
            'StreamingMovies': streamingmovies,
            'MultipleLines': multiplelines,
        
            
        }
        df = pd.DataFrame([data])
        df_principal=df.copy()
        st.dataframe(df_principal)
      
       # Preprocesamiento (codificación de variables categóricas)
        df['Type'] = df['Type'].map({'Month-to-month': 0,'One year': 1, 'Two Year': 2})
        df['PaperlessBilling'] = df['PaperlessBilling'].map({'Si': 1, 'No': 0})
        df['PaymentMethod'] = df['PaymentMethod'].map({'Credit Card': 0,'Bank Transfer':1, 'Electronick check': 2,'Mailed check':3})
        df['gender'] = df['gender'].map({'femenino': 0, 'masculino': 1})
        df['SeniorCitizen'] = df['SeniorCitizen'].map({'Si': 1, 'No': 0})
        df['Partner'] = df['Partner'].map({'Si': 1, 'No': 0})
        df['Dependents'] = df['Dependents'].map({'Si': 1, 'No': 0})
        df['InternetService'] = df['InternetService'].map({'DSL': 0, 'Fibra Optica': 1})
        df['OnlineSecurity'] = df['OnlineSecurity'].map({'Si': 1, 'No': 0})
        df['OnlineBackup'] = df['OnlineBackup'].map({'Si': 1, 'No': 0})
        df['DeviceProtection'] = df['DeviceProtection'].map({'Si': 1, 'No': 0})
        df['TechSupport'] = df['TechSupport'].map({'Si': 1, 'No': 0})
        df['StreamingTV'] = df['StreamingTV'].map({'Si': 1, 'No': 0})
        df['StreamingMovies'] = df['StreamingMovies'].map({'Si': 1, 'No': 0})
        df['MultipleLines'] = df['MultipleLines'].map({'Si': 1, 'No': 0})


     
       




        try:
            st.dataframe(df)
            # Cargar el modelo
            model = joblib.load(f"./modeling_output/model_fit/b01_model_rf.joblib")

            # Realizar la predicción
            prediction = model.predict(df)[0]

            # Mostrar el resultado
            st.subheader('Resultado de la Predicción')
            if prediction == 1:
                st.write('El cliente podría cancelar el plan.')
            else:
                st.write('El cliente seguirá con la empresa.')
        except FileNotFoundError:
            st.error("El archivo del modelo no se encontró. Verifica la ruta y el nombre del archivo.")
        except Exception as e:
            st.error(f"Ocurrió un error: {e}")
