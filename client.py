from mlflow.tracking import MlflowClient

r_models = ['RandomForestClassifier', 'XGBClassifier']
mlflow = MlflowClient()
models = []
for model_name in r_models:
    versions = mlflow.search_model_versions(f"name='{model_name}'")
    
    for version in versions:
        models.append({
            "Model Name": model_name,
            "Version": version.version,
            "Description": version.description,
        })
        run_id = version.run_id
        run = mlflow.get_run(run_id)
        metrics = run.data.metrics
        parameters = run.data.params
        models[-1]["Accuracy"] = metrics.get("accuracy", "N/A")
        models[-1]["recall"] = metrics.get("recall", "N/A")
        models[-1]['Parameters'] = parameters
        print(models[-1])