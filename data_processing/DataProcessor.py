import os
import glob
import pandas as pd


class DataProcessor:
    def __init__(self, dataset):
        self.dataset = dataset
        self.dataset.drop(columns=["tagged", "posNegObjSenti", "posNegObjSentiSum", "posNegObjSentiMean", "tags", "tagsNameChange", "text", "textSeq"],inplace=True)
        try:
            self.dataset["humor"] = self.dataset["humor"].astype(int)
        except:
            pass


    def get_processed_df(self):
        return self.dataset

