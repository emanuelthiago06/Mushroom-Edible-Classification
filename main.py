from src.dataset_treatment import pre_process
from src.model_dealing import train_and_evaluate
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier



DATA_PATH = "data/mushroom_cleaned.csv"

def run():
    df = pd.read_csv(DATA_PATH)
    df_processed = pre_process(df)
    df_processed.head()
    X = df.drop('class', axis=1)
    y = df['class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    mlp_params = [
    {'hidden_layer_sizes': (25,), 'activation': 'relu', 'solver': 'adam'},
    {'hidden_layer_sizes': (40,), 'activation': 'tanh', 'solver': 'sgd'},
    {'hidden_layer_sizes': (30, 25), 'activation': 'relu', 'solver': 'adam'}
    ]
    logistic_params = [
    {'penalty': 'l2', 'C': 1.0},
    {'penalty': 'l1', 'C': 0.5, 'solver': 'liblinear'},
    {'penalty': 'l2', 'C': 0.1}
    ]
    svm_params = [
    {'kernel': 'linear', 'C': 1.0},
    {'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale'},
    {'kernel': 'poly', 'C': 0.5, 'degree': 3}
    ]

    # Treinando # 3 algorítmos com 3 modelos diferentes cada
    print("\nResultados MLP:")
    mlp_results = train_and_evaluate(MLPClassifier, mlp_params, X_train, y_train, X_test, y_test)
    print("\nResultados Regressão Logística:")
    logistic_results = train_and_evaluate(LogisticRegression, logistic_params, X_train, y_train, X_test, y_test)
    print("\nResultados SVM:")
    svm_results = train_and_evaluate(SVC, svm_params, X_train, y_train, X_test, y_test)
    
    
    # Bagging com parâmetros padrão e customizados
    bagging_params = [
    {'random_state': 42},  # Parâmetros padrão
    {'n_estimators': 50, 'max_samples': 0.8, 'max_features': 0.8, 'random_state': 42}  # Parâmetros customizados
    ]
    print("\nResultados Bagging:")
    bagging_results = train_and_evaluate(BaggingClassifier, bagging_params, X_train, y_train, X_test, y_test)
    # RandomForest com parâmetros padrão e customizados
    rf_params = [
        {'random_state': 42},  # Parâmetros padrão
        {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 5, 'random_state': 42}  # Parâmetros customizados
    ]
    print("\nResultados RandomForest:")
    rf_results = train_and_evaluate(RandomForestClassifier, rf_params, X_train, y_train, X_test, y_test)

    # GradientBoosting
    gb_params = [
        {'random_state': 42},  # Parâmetros padrão
        {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3, 'random_state': 42}  # Parâmetros customizados
    ]

    # XGBoost
    xgb_params = [
        {'random_state': 42},  # Parâmetros padrão
        {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3, 'seed': 42}  # Parâmetros customizados
    ]

    # LightGBM
    lgb_params = [
        {'random_state': 42},  # Parâmetros padrão
        {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3, 'random_state': 42}  # Parâmetros customizados
    ]
    
    print("\nResultados GradientBoosting:")
    gb_results = train_and_evaluate(GradientBoostingClassifier, gb_params, X_train, y_train, X_test, y_test)

    print("\nResultados XGBoost:")
    xgb_results = train_and_evaluate(xgb.XGBClassifier, xgb_params, X_train, y_train, X_test, y_test)

    print("\nResultados LightGBM:")
    lgb_results = train_and_evaluate(lgb.LGBMClassifier, lgb_params, X_train, y_train, X_test, y_test)




    
    
    
if __name__ == "__main__":
    run()