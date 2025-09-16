"""
This module has been made to adapt LLMTime+Qwen (Naive Zero-Shot LLM for time series forecasting)
on this anomaly detection benchmark.

Made by Giovanni Pio Delvecchio
"""
import sys
sys.path.append("/workdir/llmtime-mod")
#sys.path.append("/content/LLMsForAnomalyDetection/llmtime-mod")
import numpy as np
import pandas as pd
import tempfile
from .base import BaseDetector

from data.serialize import SerializerSettings
from models.llmtime import get_llmtime_predictions_data

#import pdb


qwen3_hypers = dict(
    temp=1.0,
    alpha=0.99,
    beta=0.3,
    basic=False,
    settings=SerializerSettings(base=10, prec=3, time_sep=',', bit_sep='', plus_sign='', minus_sign='-', signed=True), 
)

class LLMTime(BaseDetector):
    # we keep the defaults the same as those in Chronos for compatibility
    def __init__(self, 
                 win_size=100,
                 model_size = 'base',  # [tiny, small, base]
                 prediction_length=20, 
                 input_c=1, 
                 batch_size=128):

        self.model_name = 'LLMTime'
        self.model_size = model_size
        self.win_size = win_size
        self.prediction_length = prediction_length
        self.input_c = input_c
        self.batch_size = batch_size
        self.score_list = []
        self.all_predictions = []
        self.median_values = []

    def fit(self, data):

        for channel in range(self.input_c):
            
            data_channel = data[:, channel].reshape(-1, 1)
            data_win, data_target = self.create_dataset(data_channel, slidingWindow=self.win_size, predict_time_steps=self.prediction_length)        
            #print(f"@@ Data window of shape {data_win.shape}")
            #print(data_win)
            #print(f"## Data target of shape {data_target.shape}")
            #print(data_target)
            #all_predictions = []
            for window_idx, current_series_window in enumerate(data_win):
                current_target = data_target[window_idx]
                prediction_dict = get_llmtime_predictions_data(current_series_window, current_target, 
                                                               model='qwen3-8B', settings=qwen3_hypers["settings"], 
                                                               num_samples = 10, temp=qwen3_hypers["temp"], 
                                                               alpha=qwen3_hypers["alpha"], beta=qwen3_hypers["beta"])
                predictions = prediction_dict['samples'][100].tolist() # for some reason the name of the series within the returned dataframe is 100
                self.all_predictions.append(predictions)
                median = prediction_dict['median'].tolist()[:self.prediction_length]
                self.median_values.append(median)
                #pdb.set_trace()
                

            ### using mse as the anomaly score
            scores = (data_target.squeeze() - np.array(self.median_values).squeeze()) ** 2
            self.score_list.append(scores)

        scores_merge = np.mean(np.array(self.score_list), axis=0)
        # print('scores_merge: ', scores_merge.shape)

        padded_decision_scores = np.zeros(len(data))
        padded_decision_scores[: self.win_size+self.prediction_length-1] = scores_merge[0]
        padded_decision_scores[self.win_size+self.prediction_length-1 : ]=scores_merge

        self.decision_scores_ = padded_decision_scores
    

    def decision_function(self, X):
        """ need this for compatibility """
        print("I'm a dummy function")


    def create_dataset(self, X, slidingWindow, predict_time_steps=1):
        Xs, ys = [], []
        for i in range(len(X) - slidingWindow - predict_time_steps+1):
            
            tmp = X[i : i + slidingWindow + predict_time_steps].ravel()
            # tmp= MinMaxScaler(feature_range=(0,1)).fit_transform(tmp.reshape(-1,1)).ravel()
            
            x = tmp[:slidingWindow]
            y = tmp[slidingWindow:]
            Xs.append(x)
            ys.append(y)
        return np.array(Xs), np.array(ys)

    

    


if __name__ == "__main__":
    data_direc = '/content/TSB-AD/Datasets/TSB-AD-U/001_NAB_id_1_Facility_tr_1007_1st_2014.csv'
    df = pd.read_csv(data_direc).dropna()
    data = df.iloc[:, 0:-1].values.astype(float)
    llmtime_model = LLMTime()
    llmtime_model.fit(data)