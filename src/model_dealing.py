import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve, confusion_matrix, make_scorer
import xgboost as xgb
import lightgbm as lgb

def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)

def train_and_evaluate(model_class, params_list, X, y, n_splits=10):
    results = []
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    for params in params_list:
        model = model_class(**params)
        accuracies = []
        precisions = []
        recalls = []
        specificities = []
        aucs = []

        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

            accuracies.append(accuracy_score(y_test, y_pred))
            precisions.append(precision_score(y_test, y_pred, average='weighted'))
            recalls.append(recall_score(y_test, y_pred, average='weighted'))
            specificities.append(specificity_score(y_test, y_pred))
            if y_proba is not None:
                aucs.append(roc_auc_score(y_test, y_proba))

        mean_accuracy = np.mean(accuracies)
        mean_precision = np.mean(precisions)
        mean_recall = np.mean(recalls)
        mean_specificity = np.mean(specificities)
        mean_auc = np.mean(aucs) if aucs else None
        
        results.append({
            'params': params,
            'accuracy': mean_accuracy,
            'precision': mean_precision,
            'recall': mean_recall,
            'specificity': mean_specificity,
            'auc': mean_auc
        })
        
        print(f"{model_class.__name__} Params: {params} - "
              f"Accuracy: {mean_accuracy:.4f}, Precision: {mean_precision:.4f}, "
              f"Recall: {mean_recall:.4f}, Specificity: {mean_specificity:.4f}, AUC: {mean_auc:.4f}")
    return results

# Carregar o dataset
df = pd.read_csv('caminho/para/seu/dataset.csv')

# Verificar valores nulos
print("Valores nulos em cada coluna:")
print(df.isnull().sum())

# Preencher ou remover valores nulos, conforme necessário
df = df.dropna()  # exemplo de ação, pode ser ajustado conforme o caso

# Converter a coluna de
