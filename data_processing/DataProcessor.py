import os
import glob
import pandas as pd


class DataProcessor:
    def __init__(self, data_path):
        self.path = data_path
        self.df = self.prepare_data(self.path)
        self.df = self.clean_data(self.df)


    def prepare_data(self, path):
        all_files = glob.glob(os.path.join(path, "*.csv"))
        df_from_each_file = (pd.read_csv(f,index_col=0) for f in all_files[:])
        concatenated_df   = pd.concat(df_from_each_file, ignore_index=True)

        return concatenated_df


    def clean_data(self, df):
        df.drop(columns=["text","textSeq"],inplace=True)
        df["humor"] = df["humor"].astype(int)
        
        return df


    def get_processed_df(self):
        return self.df


if __name__=='__main__':
    PATH = '.'
    dp = DataProcessor(PATH)
    df = dp.get_processed_df()

    print(df)