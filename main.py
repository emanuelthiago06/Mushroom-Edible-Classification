from src.dataset_treatment import pre_process
from src.model_dealing import train_and_evaluate
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_selection import SelectKBest, f_classif
import xgboost as xgb
import lightgbm as lgb
import mlflow
from mlflow.sklearn import log_model


DATA_PATH = "data/mushroom_cleaned.csv"


def run():
    mlflow.set_experiment("exp_projeto_ciclo_2_3.1")    
    df = pd.read_csv(DATA_PATH)
    df_processed = pre_process(df)
    df_processed.head()
    X = df.drop('class', axis=1).values
    y = df['class'].values
    # Normalizar as features
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    mlp_params = [
    {'hidden_layer_sizes': (4,), 'activation': 'relu', 'solver': 'adam', 'max_iter': 30},
    {'hidden_layer_sizes': (4,), 'activation': 'tanh', 'solver': 'sgd', 'max_iter': 30},
    {'hidden_layer_sizes': (4, 5), 'activation': 'relu', 'solver': 'adam', 'max_iter': 30}
    ]
    logistic_params = [
    {'penalty': 'l2', 'C': 1.0},
    {'penalty': 'l1', 'C': 0.5, 'solver': 'liblinear'},
    {'penalty': 'l2', 'C': 0.1}
    ]
    svm_params = [
    {'kernel': 'linear', 'C': 1.0, 'cache_size': 500},
    {'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale', 'cache_size': 500},
    {'kernel': 'poly', 'C': 0.5, 'degree': 3, 'cache_size': 500}
    ]
    linear_svc_params = [
    {'C': 1.0},
    {'C': 0.5},
    {'C': 0.1}
    ]

    # Treinando # 3 algorítmos com 3 modelos diferentes cada
    print("\nResultados MLP:")
    mlp_results = train_and_evaluate(MLPClassifier, mlp_params, X, y)
    print("\nResultados Regressão Logística:")
    logistic_results = train_and_evaluate(LogisticRegression, logistic_params, X, y)
    print("\nResultados SVM:")
    svm_results = train_and_evaluate(LinearSVC, linear_svc_params, X, y)
    
    
    # Bagging com parâmetros padrão e customizados
    bagging_params = [
    {'random_state': 42},  # Parâmetros padrão
    {'n_estimators': 50, 'max_samples': 0.8, 'max_features': 0.8, 'random_state': 42}  # Parâmetros customizados
    ]
    # RandomForest com parâmetros padrão e customizados
    rf_params = [
        {'random_state': 42},  # Parâmetros padrão
        {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 5, 'random_state': 42}  # Parâmetros customizados
    ]

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
    
    results = {}
    state = "feature_selection"
    for feature_selection in [True, None]:
        if feature_selection is not None:
            feature_selection = SelectKBest(score_func=f_classif)
            state = "_feature_selection"
        else:
            state = ""
        print("\nResultados MLP:")
        mlp_results = train_and_evaluate(MLPClassifier, mlp_params, X, y, feature_selector=feature_selection)
        print("\nResultados Regressão Logística:")
        logistic_results = train_and_evaluate(LogisticRegression, logistic_params, X, y, feature_selector=feature_selection)
        print("\nResultados SVM:")
        svm_results = train_and_evaluate(LinearSVC, linear_svc_params, X, y, feature_selector=feature_selection)
        
        print("\nResultados RandomForest:")
        rf_results = train_and_evaluate(RandomForestClassifier, rf_params, X, y, feature_selector=feature_selection)
        
        print("\nResultados Bagging:")
        bagging_results = train_and_evaluate(BaggingClassifier, bagging_params, X, y, feature_selector=feature_selection)
            
        print("\nResultados GradientBoosting:")
        gb_results = train_and_evaluate(GradientBoostingClassifier, gb_params, X, y, feature_selector=feature_selection)

        print("\nResultados XGBoost:")
        xgb_results = train_and_evaluate(xgb.XGBClassifier, xgb_params, X, y, feature_selector=feature_selection)

        print("\nResultados LightGBM:")
        lgb_results = train_and_evaluate(lgb.LGBMClassifier, lgb_params, X, y, feature_selector=feature_selection)

        results[f"mlp{state}"] = mlp_results
        results[f"logistic{state}"] = logistic_results
        results[f"svm{state}"] = svm_results
        results[f"rf{state}"] = rf_results
        results[f"bagging{state}"] = bagging_results
        results[f"gb{state}"] = gb_results
        results[f"xgb{state}"] = xgb_results
        results[f"lgb{state}"] = lgb_results
    
    results_list = []
    
    for model in results:
        for indice in results[model]:
            results_list.append(indice)
            

    df = pd.DataFrame(results_list)
    df_sorted = df.sort_values(by="accuracy", ascending=False)
    n_modelos = 3
    for i in range(n_modelos):
        model_name = df_sorted.iloc[i]['model_name']
        id = df_sorted.iloc[i]['mlflow_id']
        model_uri = f"runs:/{id}/{model_name}"
        model_name = f"{model_name}"

        mlflow.register_model(model_uri=model_uri, name=model_name)
        print(f"Modelo registrado com sucesso, modelo: {model_name} de id: {id}")
    
    
    
if __name__ == "__main__":
    run()