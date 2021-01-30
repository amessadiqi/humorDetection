import pandas as pd
from flask import Flask, request
from humor_features.HumorFeatures import HumorFeatures
from data_processing.DataProcessor import DataProcessor
from humor_model.Models import Models


app = Flask(__name__)


@app.route('/predict/<method>/<text>', methods=['GET'])
def predict(method, text):
    data = pd.DataFrame({"text": text}, index=["text"])
    X = DataProcessor(
        HumorFeatures(data).getStructure().getFreq().getWrittenSpoken().getSyno().getSynsets().getSentiment().df
    ).get_processed_df()

    res = Models().predict(method=method , val=X)

    if int(res[0]):
        result = True
    else:
        result = False

    return {
        "status": "success",
        "humor": result
    }


@app.route('/methods/', methods=['GET'])
def get_methods():
    return {
        "methods": [
            "XGBoost",
            "Logistic Regression",
            "Random Forests"
        ]
    }
