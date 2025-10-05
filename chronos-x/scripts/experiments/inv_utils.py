from gluonts.dataset.common import ListDataset
import pandas as pd
from datasets import Dataset
from gluonts.dataset.split import split
from chronosx.utils.hf_data_loader import to_gluonts_univariate
import pickle

import pdb

def create_rolling_dataset_with_covariates(gluonts_series, window_size=100, prediction_length=1):
    """
    Trasforma una serie GluonTS in un dataset rolling window con covariate dinamiche.
    
    Args:
        gluonts_series: lista di dict con 'target', 'start' e opzionale 'feat_dynamic_real'
        window_size: lunghezza della finestra di contesto
        prediction_length: lunghezza della predizione (di solito 1)
        
    Returns:
        ListDataset pronto per ChronosX/forecasting autoregressivo
    """
    new_data = []

    for ts in gluonts_series:
        target = ts['target']
        start = ts['start']
        covariates = ts.get('feat_dynamic_real', None)

        for i in range(len(target) - window_size - prediction_length + 1):
            window = target[i:i+window_size]
            future = target[i+window_size:i+window_size+prediction_length]

            item = {
                'target': window,
                'future_target': future,
                'start': start  # puoi aggiustare la data se vuoi
            }

            # se ci sono covariate dinamiche, estrai lo stesso intervallo
            if covariates is not None:
                # shape: (num_covariates, time)
                #past_covariates = covariates[:, i:i+window_size]
                window_covariates = covariates[:, i:i+window_size+prediction_length]
                # window_covariates[window_covariates == 0] = 0.5
                #future_covariates = covariates[:, i+window_size:i+window_size+prediction_length]
                item['feat_dynamic_real'] = window_covariates
                #item['future_covariates'] = future_covariates

            new_data.append(item)

    # freq la prende dalla prima serie
    freq = getattr(gluonts_series[0]['start'], 'freqstr', None)
    return ListDataset(new_data, freq=freq)


def split_rolling_dataset(train_p, val_p, test_p, complete_dataset):
    total_len = len(complete_dataset)
    train_p_index = (total_len * train_p) // 100
    val_p_index = train_p_index + ((total_len * val_p) // 100)
    test_p_index = train_p_index + val_p_index + ((total_len * test_p) // 100)

    rolling_train_dataset = complete_dataset[:train_p_index]
    rolling_val_dataset = complete_dataset[train_p_index:val_p_index]
    rolling_test_dataset = complete_dataset[val_p_index:test_p_index]
    return rolling_train_dataset, rolling_val_dataset, rolling_test_dataset


def add_dates_and_convert_to_gluonts(dataframe, target_col, add_other_cols=True) -> pd.DataFrame:
    freq = "1D"
    T = dataframe.shape[0]
    date_range = pd.date_range("2025-01-01", periods=T, freq=freq)

    ts = pd.DataFrame(index=date_range)
    ts["target"] = dataframe[target_col].values

    if add_other_cols:
        for col in dataframe.columns:
            if col != target_col:
                # copia i valori e sostituisci 0 con 0.5
                ts[col] = dataframe[col].replace(0, 0.5).values
    return ts


def load_anomaly_dataset(anomaly_file = 
                         '/workdir/LLMsForAnomalyDetection/TSB-AD/Datasets/TSB-AD-U/001_NAB_id_1_Facility_tr_1007_1st_2014.csv'):
    dataset_name = "anomaly"
    prediction_length = 1
    num_covariates = 2

    print("@@ LOADING DATASET @@")
    anomaly_df = pd.read_csv(anomaly_file)
    gluonts_anomaly_df = add_dates_and_convert_to_gluonts(anomaly_df, "Data", add_other_cols=True)

    anomaly_hf_dataset = Dataset.from_list([{
        "timestamp": gluonts_anomaly_df.index.astype(str).tolist(),
        "target": gluonts_anomaly_df["target"].tolist(),
        "Label": gluonts_anomaly_df["Label"].tolist(),
    }])
    anomaly_hf_dataset.set_format("numpy")

    series_fields = ["target"]
    covariates_fields = ["Label"]
    
    gts_dataset = to_gluonts_univariate(anomaly_hf_dataset, series_fields, covariates_fields)
    
    #anomaly_train_dataset, anomaly_test_template = split(gts_dataset, offset=-1) # only leaves the last entry in the test set
    #dataset_len = gts_dataset[0]["target"].shape[0]
    #anomaly_train_dataset, anomaly_test_template = split(gts_dataset, offset=dataset_len // 2) # first 3000 entries in the training set
    #anomaly_test_dataset = anomaly_test_template.generate_instances(prediction_length, windows=dataset_len // 2, distance=1)


    rolling_dataset = create_rolling_dataset_with_covariates(
        gts_dataset,  # o la lista di dict GluonTS
        window_size=100,
        prediction_length=1
    )

    rolling_train_dataset, rolling_val_dataset, rolling_test_dataset = split_rolling_dataset(80, 10, 10, rolling_dataset)

    return rolling_train_dataset, rolling_val_dataset, rolling_test_dataset


def load_inverse_opt_results(file_path):
    """
    Carica tutti i batch salvati in append nel file pickle.
    Restituisce una lista di dizionari, uno per batch.
    """
    results = []
    with open(file_path, "rb") as f:
        while True:
            try:
                batch_result = pickle.load(f)
                results.append(batch_result)
            except EOFError:
                break
    return results


if __name__ == "__main__":
    """
    file_path = "inverse_opt_results_stream.pkl"
    all_batches = load_inverse_opt_results(file_path)

    opt_past_covariates = []
    opt_future_covariates = []
    for elem in all_batches:
        opt_past_covariates.append(elem["past_covariates_final"])
        opt_future_covariates.append(elem["future_covariates_final"])

    import pdb
    pdb.set_trace()
    """


    train, val, test = load_anomaly_dataset()

    pdb.set_trace()

