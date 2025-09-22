""" If this demo doesn't work because of jax, use the following:

    pip install -U "jax[cuda12]"
"""


import os
import torch
os.environ['OMP_NUM_THREADS'] = '4'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import openai
#openai.api_key = os.environ['OPENAI_API_KEY']
#openai.api_base = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
from data.serialize import SerializerSettings
from models.utils import grid_iter
from models.promptcast import get_promptcast_predictions_data
from models.darts import get_arima_predictions_data
from models.llmtime import get_llmtime_predictions_data
from data.small_context import get_datasets
from models.validation_likelihood_tuning import get_autotuned_predictions_data

from extract_ts_from_stock_file import extract_ts_with_range

def plot_preds(train, test, pred_dict, model_name, show_samples=False):
    pred = pred_dict['median']
    pred = pd.Series(pred, index=test.index)
    plt.figure(figsize=(8, 6), dpi=100)
    plt.plot(train)
    plt.plot(test, label='Truth', color='black')
    plt.plot(pred, label=model_name, color='purple')
    # shade 90% confidence interval
    samples = pred_dict['samples']
    lower = np.quantile(samples, 0.05, axis=0)
    upper = np.quantile(samples, 0.95, axis=0)
    plt.fill_between(pred.index, lower, upper, alpha=0.3, color='purple')
    if show_samples:
        samples = pred_dict['samples']
        # convert df to numpy array
        samples = samples.values if isinstance(samples, pd.DataFrame) else samples
        for i in range(min(10, samples.shape[0])):
            plt.plot(pred.index, samples[i], color='purple', alpha=0.3, linewidth=1)
    plt.legend(loc='upper left')
    if 'NLL/D' in pred_dict:
        nll = pred_dict['NLL/D']
        if nll is not None:
            plt.text(0.03, 0.85, f'NLL/D: {nll:.2f}', transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5))
    plt.savefig(f"{model_name}_outputplot.png")
    plt.close()


print(torch.cuda.max_memory_allocated())
print()

gpt4_hypers = dict(
    alpha=0.3,
    basic=True,
    temp=1.0,
    top_p=0.8,
    settings=SerializerSettings(base=10, prec=3, signed=True, time_sep=', ', bit_sep='', minus_sign='-')
)

mistral_api_hypers = dict(
    alpha=0.3,
    basic=True,
    temp=1.0,
    top_p=0.8,
    settings=SerializerSettings(base=10, prec=3, signed=True, time_sep=', ', bit_sep='', minus_sign='-')
)

gpt3_hypers = dict(
    temp=0.7,
    alpha=0.95,
    beta=0.3,
    basic=False,
    settings=SerializerSettings(base=10, prec=3, signed=True, half_bin_correction=True)
)

"""
llma2_hypers = dict(
    temp=0.7,
    alpha=0.95,
    beta=0.3,
    basic=False,
    settings=SerializerSettings(base=10, prec=3, signed=True, half_bin_correction=True)
)
"""

llama_hypers = dict(
    temp=1.0,
    alpha=0.99,
    beta=0.3,
    basic=False,
    settings=SerializerSettings(base=10, prec=3, time_sep=',', bit_sep='', plus_sign='', minus_sign='-', signed=True), 
)


promptcast_hypers = dict(
    temp=0.7,
    settings=SerializerSettings(base=10, prec=0, signed=True, 
                                time_sep=', ',
                                bit_sep='',
                                plus_sign='',
                                minus_sign='-',
                                half_bin_correction=False,
                                decimal_point='')
)

arima_hypers = dict(p=[12,30], d=[1,2], q=[0])

mistral_hf_hypers = dict(
    temp=0.7,
    alpha=0.95,
    beta=0.3,
    basic=False,
    settings=SerializerSettings(
        base=10, prec=3, signed=True, half_bin_correction=True
    )
)

llama_hypers = dict(
    temp=1.0,
    alpha=0.99,
    beta=0.3,
    basic=False,
    settings=SerializerSettings(base=10, prec=3, time_sep=',', bit_sep='', plus_sign='', minus_sign='-', signed=True), 
)

qwen3_hypers = dict(
    temp=1.0,
    alpha=0.99,
    beta=0.3,
    basic=False,
    settings=SerializerSettings(base=10, prec=3, time_sep=',', bit_sep='', plus_sign='', minus_sign='-', signed=True), 
)

model_hypers = {
     #'LLMTime GPT-3.5': {'model': 'gpt-3.5-turbo-instruct', **gpt3_hypers},
     #'LLMTime GPT-4': {'model': 'gpt-4', **gpt4_hypers},
     #'LLMTime GPT-3': {'model': 'text-davinci-003', **gpt3_hypers},
     #'PromptCast GPT-3': {'model': 'text-davinci-003', **promptcast_hypers},
     'LLAMA2': {'model': 'llama-7b', **llama_hypers},
     'qwen3': {'model': 'qwen3', **qwen3_hypers},
     #'mistral': {'model': 'mistral', **llma2_hypers},
     #'mistral-api-tiny': {'model': 'mistral-api-tiny', **mistral_api_hypers},
     #'mistral-api-small': {'model': 'mistral-api-tiny', **mistral_api_hypers},
     #'mistral-api-medium': {'model': 'mistral-api-tiny', **mistral_api_hypers},
     'Mistral-7B': {'model': 'mistralai/Mistral-7B-v0.3', **mistral_hf_hypers}, 
     'Mixtral-8x7B': {'model': 'mistralai/Mixtral-8x7B-v0.1', **mistral_hf_hypers},  
     'ARIMA': arima_hypers
 }

"""
model_predict_fns = {
    'LLMA2': get_llmtime_predictions_data,
    #'mistral': get_llmtime_predictions_data,
    #'LLMTime GPT-4': get_llmtime_predictions_data,
    #'mistral-api-tiny': get_llmtime_predictions_data
}
"""

model_predict_fns = {
    #'LLAMA2': get_llmtime_predictions_data,
    'qwen3': get_llmtime_predictions_data,
    #'Mistral-7B': get_llmtime_predictions_data,
    #'Mixtral-8x7B': get_llmtime_predictions_data,
    #'ARIMA': get_arima_predictions_data,
}


model_names = list(model_predict_fns.keys())

print(model_hypers.keys())

#datasets = get_datasets()
#ds_name = 'AirPassengersDataset'


#data = datasets[ds_name]
#train, test = data # or change to your own data

ds_name = "Apple Stocks"
df = pd.read_csv("./sp225.csv")
symbol = "AAPL"

# Estraggo i dati nel range
extracted_col, dates = extract_ts_with_range(symbol, df, "2021", "2024")

# Creo una serie con indice = date e nome 'Month'
series = pd.Series(extracted_col.values, index=dates)
series.index.name = "Month"

# Split train/test
test_idx = int(np.floor(series.shape[0] - 0.2 * series.shape[0]))
train = series.iloc[:test_idx]
test = series.iloc[test_idx:]

#print("Train sample:")
#print(train.head())
#print("\nTest sample:")
#print(test.head())

#print(type(train))
#print(train.iloc[0:10])

out = {}

for model in model_names: # GPT-4 takes a about a minute to run
    model_hypers[model].update({'dataset_name': ds_name}) # for promptcast
    hypers = list(grid_iter(model_hypers[model]))
    num_samples = 10
    print(train)
    print(test)
    pred_dict = get_autotuned_predictions_data(train, test, hypers, num_samples, model_predict_fns[model], verbose=False, parallel=False)
    #print(pred_dict)
    out[model] = pred_dict
    #plot_preds(train, test, pred_dict, model, show_samples=True)
