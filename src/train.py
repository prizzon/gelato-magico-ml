import os
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

DATA_PATH = os.path.join("data", "sorvete.csv")
EXPERIMENT_NAME = "GelatoMagico-PrevisaoVendas"

def main():
    df = pd.read_csv(DATA_PATH)

    X = df[["temperatura"]]
    y = df["vendas"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    model = LinearRegression()

    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run():
        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        rmse = mse ** 0.5
        r2 = r2_score(y_test, preds)

        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_metric("MAE", float(mae))
        mlflow.log_metric("RMSE", float(rmse))
        mlflow.log_metric("R2", float(r2))

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="GelatoMagicoModel"
        )

        print("Treinamento concluído!")
        print(f"MAE={mae:.2f} | RMSE={rmse:.2f} | R2={r2:.4f}")

if __name__ == "__main__":
    main()
