import pandas as pd
from data_processing.DataProcessor import DataProcessor
from humor_features.HumorFeatures import HumorFeatures
from humor_model.Models import Models
from prediction_server.app import app


class HumorDetector:
    def __init__(self, dataset = None):
        if isinstance(dataset, pd.DataFrame):
            print('Calculating the features...')
            self.dataset = HumorFeatures(dataset).getAllFeatures()
            print('Processing the dataset...')
            self.dataset = DataProcessor(dataset=self.dataset).get_processed_df()


    def get_processed_dataset(self):
        return self.dataset


    def predict(self , val , method):
        data = pd.DataFrame({"text": val}, index=["text"])
        X = DataProcessor(
            HumorFeatures(data).getAllFeatures()
        ).get_processed_df()
            
        return Models().predict(method=method , val=X)


    def performance_overview(self):
        Models(dataset=self.dataset).displayScores()


    def prediction_server(self, host, port):
        app.run(host=host, port=port)
