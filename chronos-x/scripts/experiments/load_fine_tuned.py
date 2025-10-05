import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from transformers import AutoConfig

from chronosx.chronosx import ChronosXPipeline
from chronosx.utils.chronos_dataset import ChronosDataset

from inv_utils import load_anomaly_dataset
from gluonts.model.evaluation import evaluate_forecasts
from gluonts.ev.metrics import MASE, MeanWeightedSumQuantileLoss
from gluonts.dataset.split import TestData, OffsetSplitter

import pdb

# -------------------- Main --------------------

print("@@ START OF THE SCRIPT @@")
lr = 0.0001
run_id=1
MODEL_CHECKPOINT = f"../../output/finetune/anomaly/lr={lr}/run_id={run_id}/final-checkpoint"
INJECTION_METHOD = "IIB+OIB"

dataset_name = "anomaly"
prediction_length = 1
num_covariates = 2

train_dataset, test_dataset, rolling_test_dataset = load_anomaly_dataset()

print("@@ LOADING MODEL @@")
chronos_config = AutoConfig.from_pretrained(MODEL_CHECKPOINT)
chronos_model_id = "amazon/chronos-t5-small"

pipeline = ChronosXPipeline(
    prediction_length=prediction_length,
    num_covariates=num_covariates,
    covariate_injection=INJECTION_METHOD,
    pretrained_model_name_or_path=MODEL_CHECKPOINT,
    )

tokenizer = pipeline.tokenizer
chronosx_model = pipeline.chronosx


quantized_val_dataset = ChronosDataset(
    datasets=[train_dataset],
    probabilities=[1.0],
    tokenizer=tokenizer,
    prediction_length=prediction_length,
    min_past=1,
    mode="validation",
)

print("@@ RUNNING MANUAL EVALUATION @@")
chronosx_model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
chronosx_model.to(device)

mse_losses, mae_losses = [], []
save_dir = Path("./plots")


pipeline.chronosx.eval()

splitter = OffsetSplitter(offset=-prediction_length)
test_wrapper = TestData(test_dataset, splitter=splitter, prediction_length=prediction_length)
forecasts = pipeline.generate_forecasts(
    test_wrapper.input,
)

metrics = (
    evaluate_forecasts(
        forecasts,
        test_data=test_wrapper,
        metrics=[
            MASE(),
            MeanWeightedSumQuantileLoss(np.arange(0.05, 1, 0.05).round(2).tolist()),
        ],
        batch_size=5000,
    )
    .reset_index(drop=True)
    .to_dict(orient="records")
)

# ------------------ TEACHER FORCING ------------------
"""
with torch.no_grad():
    for i, batch in enumerate(quantized_val_dataset):
        context = torch.tensor(batch["input_ids"]).unsqueeze(0).to(device)
        past_covariates = torch.tensor(batch["past_covariates"]).unsqueeze(0).to(device)
        future_covariates = torch.tensor(batch["future_covariates"]).unsqueeze(0).to(device)
        labels_tokens = torch.tensor(batch["labels"]).unsqueeze(0).to(device).long()

        
        outputs = chronosx_model(
            input_ids=context,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            labels=labels_tokens,
        )

        res = compute_ce_and_regression_losses(outputs, labels_tokens, tokenizer, scale=batch["scale"])

        mse_losses.append(res["mse"].item())
        mae_losses.append(res["mae"].item())

        print(f"[Batch {i}] CE={res['ce']:.4f} MSE={res['mse']:.4f} MAE={res['mae']:.4f}")

        if i < 5:
            plot_predictions(res["pred_values"][0], res["true_values"][0], i, save_dir=save_dir)

mean_mse = np.nanmean(mse_losses)
mean_mae = np.nanmean(mae_losses)
print(f"@@ DONE - Mean MSE={mean_mse:.4f}, Mean MAE={mean_mae:.4f} @@")
"""

# ------------------ AUTOREGRESSIVE ------------------ 

"""
with torch.no_grad():
    for i, batch in enumerate(quantized_val_dataset):
        context = torch.tensor(batch["input_ids"]).unsqueeze(0).to(device)
        past_covariates = torch.tensor(batch["past_covariates"]).unsqueeze(0).to(device)
        future_covariates = torch.tensor(batch["future_covariates"]).unsqueeze(0).to(device)
        labels_tokens = torch.tensor(batch["labels"]).unsqueeze(0).to(device).long()

        # usa generate invece del loop manuale
        pred_ids = chronosx_model.generate(
            input_ids=context,
            #max_length=context.size(1) + prediction_length,
            max_new_tokens=prediction_length,
            num_beams=1,                     # greedy
            do_sample=False,                 # disattiva sampling
            past_covariates=past_covariates,
            future_covariates=future_covariates
            #decoder_start_token_id=tokenizer.pad_token_id,  # o eos, dipende dal modello
        )

        if pred_ids.size(1) > prediction_length:
            pred_ids = pred_ids[:, 1:]

        # 2) Rimuovi EOS dalle labels
        labels_tokens = labels_tokens[:, :-1]

        # 3) Ora devono avere la stessa shape
        assert pred_ids.shape == labels_tokens.shape, \
            f"Shape mismatch: pred {pred_ids.shape}, labels {labels_tokens.shape}"


        # calcolo delle loss
        res = compute_losses_autoregressive(pred_ids, labels_tokens, tokenizer, scale=batch["scale"])
        print(f"[Batch {i}] MSE={res['mse']:.4f} MAE={res['mae']:.4f}")
"""



mse_losses, mae_losses = [], []


all_medians = []
all_lows = []
all_highs = []

all_true = []

for i, (forecast, item) in enumerate(zip(forecasts, rolling_test_dataset)):
    # forecast.samples shape: [num_samples, prediction_length]
    # di solito num_samples = 20 o simili, qui prendiamo la media
    pred_values = forecast.samples.mean(axis=0)  # shape: (prediction_length,)

    # ground truth
    true_values = item['future_target']  # shape: (prediction_length,)

    # converti in torch
    pred_tensor = torch.tensor(pred_values, dtype=torch.float32)
    true_tensor = torch.tensor(true_values, dtype=torch.float32)

    # mask per NaN
    mask = ~torch.isnan(true_tensor)
    if mask.sum() == 0:
        mse_losses.append(float('nan'))
        mae_losses.append(float('nan'))
    else:
        mse = F.mse_loss(pred_tensor[mask], true_tensor[mask]).item()
        mae = F.l1_loss(pred_tensor[mask], true_tensor[mask]).item()
        mse_losses.append(mse)
        mae_losses.append(mae)

    # saving data for future plots
    samples = forecast.samples
    median = np.median(samples, axis=0)
    low = np.percentile(samples, 10, axis=0)   # 10th percentile → basso
    high = np.percentile(samples, 90, axis=0)  # 90th percentile → alto

    all_medians.extend(median)
    all_lows.extend(low)
    all_highs.extend(high)

    all_true.append(true_values) 

save_dir.mkdir(exist_ok=True, parents=True)






# indici per la forecast (dopo la parte storica)
x_axis = np.arange(100, 200, 1)

plt.figure(figsize=(12,5))
plt.plot(x_axis, all_true[0: 100], color="royalblue", label="Historical Data")
plt.plot(x_axis, all_medians[0:100], color="tomato", label="Median Forecast")
plt.fill_between(x_axis, all_lows[0:100], all_highs[0:100], color="tomato", alpha=0.3, label="80% Prediction Interval")
plt.xlabel("Time Steps")
plt.ylabel("Value")
plt.title("Time Series Forecast with Rolling Windows")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(save_dir / "full_timeseries_forecast.png")
plt.close()
    

# statistiche complessive
mean_mse = torch.tensor(mse_losses).nanmean().item()
mean_mae = torch.tensor(mae_losses).nanmean().item()
print(f"Mean MSE={mean_mse:.4f}, Mean MAE={mean_mae:.4f}")