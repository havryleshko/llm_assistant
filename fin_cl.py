from pycaret.classification import load_model, predict_model
import pandas as pd
import warnings
import os
warnings.filterwarnings("ignore")

def predict_burn(file_path: str): # file_path = path to .csv or .xls upload

    #loading file
    ext = os.path.splitext()(file_path)[-1].lower()
    if ext == '.csv':
        loader = pd.read_csv(file_path)
    elif ext in ['.xls', '.xlsx']:
        loader = pd.read_excel(file_path) 
    else: 
        raise ValueError(f"Unsupported file type: '{ext}'")

    model = load_model("burn_cash_model.pkl")

    predictions = predict_model(model, data=loader)

    return predictions
