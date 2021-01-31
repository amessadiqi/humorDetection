import pandas as pd
from data_processing.DataProcessor import DataProcessor
from humor_features.HumorFeatures import HumorFeatures
from humor_model.Models import Models
from prediction_server.app import app


class HumorDetector:
    def __init__(self, dataset = None):
        if dataset != None:
            self.dataset = HumorFeatures(dataset).getStructure().getFreq().getWrittenSpoken().getSyno().getSynsets().getSentiment()
            self.dataset = DataProcessor(dataset=self.dataset.df).get_processed_df()


    def get_processed_dataset(self):
        return self.dataset


    def predict(self , val , method):
        data = pd.DataFrame({"text": val}, index=["text"])
        X = DataProcessor(
            HumorFeatures(data).getStructure().getFreq().getWrittenSpoken().getSyno().getSynsets().getSentiment().df
        ).get_processed_df()
            
        return Models().predict(method=method , val=X)


    def train(self):
        pass

    def performance_overview(self):
        Models(dataset=self.dataset).displayScores()


    def prediction_server(self, host, port):
        app.run(host=host, port=port)


if __name__=='__main__':
    HumorDetector().prediction_server(host='0.0.0.0', port=5000)
