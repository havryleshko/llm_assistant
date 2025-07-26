### FINANCIAL MODEL TO WORK WITH FIN DATA
# 1. loads fin model
# 2. preprocesses fin data
# 3. returns predictions

from pycaret.classification import load_model, predict_model
import pandas as pd
import warnings
import os
from FE import feature_eng

warnings.filterwarnings("ignore")

def predict_burn(df: str): # file_path = path to .csv or .xls upload
    predictions = predict_model(model, data=df)
    processes_df = feature_eng(df)
    X = processes_df.drop(['company' 'year', 'burn_cash'], axis=1)
    model = load_model("burn_cash_model.pkl")
    predictions = predict_model(model, data=X)
    return predictions
