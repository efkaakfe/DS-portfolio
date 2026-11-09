import argparse
import json
import os
import pickle
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import recall_score
from IPython.display import display

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / 'artifacts' / 'prob' / 'config_inference_prob.json'
DATA_PATH = PROJECT_ROOT / 'data' / 'test.csv'
INFER_PATH = PROJECT_ROOT / 'inference'

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def inference(config_path=CONFIG_PATH, data_path=DATA_PATH):

    config = load_config(config_path)
    
    model_name = config.get('model_name')      
    method = config.get('method')              
    threshold = config.get('threshold', 0.5)   

    data_transformer_path = Path(config.get('data_transformer'))
    model_path = Path(config.get('model_path'))
    features = config.get('features', None)    

    print(f"--- Running inference for: {model_name} (method: {method}) ---")

    df = pd.read_csv(data_path)
    
    if 'Response' in df.columns:
        y_true = df['Response'].values
        X = df.drop(columns=['Response'])
    else:
        X = df.copy()
        y_true=None

    with open(data_transformer_path, 'rb') as f:
        data_transformer = pickle.load(f)

    X_processed = data_transformer.transform(X)

    if features is not None:
        X_processed = X_processed.iloc[:, features]

    if model_name in ['CatBoost', 'LightGBM']:

        with open(model_path, 'rb') as f:
            model = pickle.load(f)
            probs = model.predict_proba(X_processed)[:, 1]
            preds = (probs >= threshold).astype(int)

    elif model_name == 'MLP':

        mlp = torch.load(model_path, weights_only=False)
        preds = mlp.predict(X_processed)

    else:
        raise ValueError(f"Model not recognized: {model_name}")

    if y_true is not None:
        score = recall_score(y_true, preds)

        results_df = pd.DataFrame({
            'Model': [model_name],
            'Method': [method],
            'Threshold': [threshold],
            'Recall_Score': [score]
        })

        display(results_df)
    
    infer_path = os.path.join(INFER_PATH, method)

    os.makedirs(infer_path, exist_ok=True)

    pred_output = pd.DataFrame({'prediction': preds})
    if y_true is not None:
        pred_output.insert(0, 'true_label', y_true)
    pred_output.to_csv(os.path.join(infer_path,'predictions.csv'), index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run inference using a saved model configuration.')
    parser.add_argument('--config', type=Path, default=CONFIG_PATH, help='Path to config_inference_*.json')
    parser.add_argument('--data', type=Path, default=DATA_PATH, help='CSV file containing input features')
    args = parser.parse_args()
    inference(args.config, args.data)
