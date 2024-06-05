import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

def pre_process(df, class_name = "class") -> pd.DataFrame:
    # Mostrar as primeiras linhas do dataframe
    print("Primeiras linhas do dataframe:")
    print(df.head())

    # Verificar valores nulos
    print("\nValores nulos em cada coluna:")
    print(df.isnull().sum())

    # Informações gerais do dataframe
    print("\nInformações gerais do dataframe:")
    print(df.info())

    # Descrição estatística das features
    print("\nDescrição estatística das features:")
    print(df.describe())

    # Calcular a correlação entre as features
    print("\nMatriz de correlação:")
    correlation_matrix = df.corr()
    print(correlation_matrix)


    # Normalizar as features
    scaler = MinMaxScaler()
    df[df.columns] = scaler.fit_transform(df[df.columns])

    print("\nDataframe após normalização:")
    print(df.head())
    return df
