
from TSB_AD.models.LLMTime import LLMTime

"""
from TSB_AD.models.Chronos import Chronos
import pandas as pd

if __name__ == "__main__":
    data_direc = './TSB-AD/Datasets/TSB-AD-U/001_NAB_id_1_Facility_tr_1007_1st_2014.csv'
    df = pd.read_csv(data_direc).dropna()
    data = df.iloc[:, 0:-1].values.astype(float)
    llmtime_model = LLMTime()
    llmtime_model.fit(data)
"""




import pandas as pd
from TSB_AD.model_wrapper import run_Unsupervise_AD
from TSB_AD.evaluation.metrics import get_metrics
import numpy as np
import pdb

# Specify Anomaly Detector to use and data directory
AD_Name = 'LLMTime'   # It can be replaced with any anomaly detector availale in TSB-AD
data_direc = './TSB-AD/Datasets/TSB-AD-U/001_NAB_id_1_Facility_tr_1007_1st_2014.csv'

# Loading Data
df = pd.read_csv(data_direc).dropna()
data = df.iloc[:, 0:-1].values.astype(float)
label = df['Label'].astype(int).to_numpy()

first_anomaly_index = np.where(label == 1)[0][0]
print(first_anomaly_index)
data = data[first_anomaly_index-100:first_anomaly_index+5]
label = label[first_anomaly_index-100:first_anomaly_index+5]

# Applying Anomaly Detector
output = run_Unsupervise_AD(AD_Name, data)
#pdb.set_trace()

# Evaluation
evaluation_result = get_metrics(output, label)
print(evaluation_result)