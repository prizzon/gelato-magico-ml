# 🍦 Gelato Mágico — Previsão de Vendas com Machine Learning + MLflow

Este projeto cria um modelo de **regressão** para prever a quantidade de sorvetes vendidos com base na **temperatura do dia**, ajudando a sorveteria **Gelato Mágico** a planejar produção e reduzir desperdício.

## 🎯 Objetivos
- Treinar um modelo para prever vendas a partir da temperatura
- Registrar métricas e versão do modelo usando **MLflow**
- Disponibilizar o modelo para previsão em “tempo real” via **API (FastAPI)**
- Garantir reprodutibilidade com pipeline simples (scripts + tracking)

## 🧠 Dataset
Arquivo: `data/sorvete.csv`  
Colunas:
- `temperatura` (°C)
- `vendas` (quantidade de sorvetes)

## 🧪 MLflow (tracking e versionamento)
O treinamento registra no MLflow:
- Parâmetros: tipo do modelo
- Métricas: MAE, RMSE e R²
- Artefato: modelo treinado (registrado como `GelatoMagicoModel`)

## 🚀 Como executar
```bash
pip install -r requirements.txt
mlflow ui
python src/train.py
python src/predict.py
uvicorn src.api:app --reload
