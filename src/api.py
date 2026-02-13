import mlflow
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

EXPERIMENT_NAME = "GelatoMagico-PrevisaoVendas"

app = FastAPI(title="Gelato Mágico - Previsão de Vendas")

class PredictRequest(BaseModel):
    temperatura: float

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

model = None

@app.on_event("startup")
def startup_event():
    global model
    model = load_latest_model()

@app.post("/predict")
def predict(req: PredictRequest):
    X = pd.DataFrame({"temperatura": [req.temperatura]})
    pred = model.predict(X)[0]
    return {"temperatura": req.temperatura, "vendas_previstas": float(pred)}

