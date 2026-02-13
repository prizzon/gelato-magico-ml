import mlflow
import pandas as pd

EXPERIMENT_NAME = "GelatoMagico-PrevisaoVendas"

def load_latest_model():
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        raise RuntimeError("Experimento não encontrado. Rode o train.py primeiro.")

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=1
    )
    if not runs:
        raise RuntimeError("Nenhum run encontrado. Rode o train.py primeiro.")

    run_id = runs[0].info.run_id
    model_uri = f"runs:/{run_id}/model"
    return mlflow.sklearn.load_model(model_uri)

def main():
    model = load_latest_model()

    temp = float(input("Informe a temperatura (°C): "))
    X = pd.DataFrame({"temperatura": [temp]})
    pred = model.predict(X)[0]

    print(f"Temperatura: {temp:.1f}°C -> Previsão de vendas: {pred:.0f} sorvetes")

if __name__ == "__main__":
    main()

