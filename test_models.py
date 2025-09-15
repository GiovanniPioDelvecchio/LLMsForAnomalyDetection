from TSB_AD.models.LLMTime import LLMTime
from TSB_AD.models.Chronos import Chronos
import pandas as pd

if __name__ == "__main__":
    data_direc = './TSB-AD/Datasets/TSB-AD-U/001_NAB_id_1_Facility_tr_1007_1st_2014.csv'
    df = pd.read_csv(data_direc).dropna()
    data = df.iloc[:, 0:-1].values.astype(float)
    llmtime_model = LLMTime()
    llmtime_model.fit(data)