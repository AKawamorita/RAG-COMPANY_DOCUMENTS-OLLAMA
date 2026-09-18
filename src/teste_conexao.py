import mlflow

# Aponta para a instância do MLflow Server recém-iniciado
mlflow.set_tracking_uri("http://localhost:5000")

mlflow.set_experiment("teste-conexao-mysql")

with mlflow.start_run():
    mlflow.log_param("parametro_teste", 123)
    mlflow.log_metric("acuracia", 0.99)
    print("Run registrada com sucesso!")