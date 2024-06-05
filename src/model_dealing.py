from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

def train_and_evaluate(model_class, params_list, X_train, y_train, X_test, y_test):
    results = []
    for params in params_list:
        model = model_class(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results.append((params, accuracy))
        print(f"{model_class.__name__} Params: {params} - Accuracy: {accuracy:.4f}")
        print(classification_report(y_test, y_pred))
    return results