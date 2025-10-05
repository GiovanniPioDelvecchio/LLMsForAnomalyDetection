import numpy as np
import matplotlib.pyplot as plt
from inv_utils import load_anomaly_dataset
from transformers import AutoConfig
from chronosx.chronosx import ChronosXPipeline
from chronosx.utils.chronos_dataset import ChronosDataset
from inv_utils import load_inverse_opt_results
import torch


from gluonts.model.evaluation import evaluate_forecasts
from gluonts.ev.metrics import MASE, MeanWeightedSumQuantileLoss
from gluonts.dataset.split import TestData, OffsetSplitter

import pdb

def plot_preds_vs_groundtruth(all_true, all_medians, all_lows, all_highs, save_dir):
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
    plt.savefig(save_dir)
    plt.close()


def plot_future_covariates(no_opt_future_covariates, future_covariates, save_dir):

    # converto in numpy
    no_opt_np = no_opt_future_covariates[0].cpu().numpy()  # shape (2,2)
    opt_np = future_covariates[0].cpu().numpy()           # shape (2,2)

    labels = ["t0_f1", "t0_f2", "t1_f1", "t1_f2"]

    # flatten per plot a barre
    no_opt_flat = no_opt_np.flatten()
    opt_flat = opt_np.flatten()

    # crea due grafici uno sopra l'altro
    fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    axs[0].bar(labels, no_opt_flat, color='gray')
    axs[0].set_title("No-Optimization Future Covariates")
    axs[0].set_ylim(min(opt_flat.min(), no_opt_flat.min()) - 0.5,
                    max(opt_flat.max(), no_opt_flat.max()) + 0.5)

    axs[1].bar(labels, opt_flat, color='blue')
    axs[1].set_title("Optimized Future Covariates")
    axs[1].set_ylim(min(opt_flat.min(), no_opt_flat.min()) - 0.5,
                    max(opt_flat.max(), no_opt_flat.max()) + 0.5)

    plt.tight_layout()
    plt.savefig(save_dir)
    plt.close()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_name = "anomaly"
    prediction_length = 1
    num_covariates = 2
    train_dataset, val_dataset, test_dataset = load_anomaly_dataset()

    print("@@ LOADING MODEL @@")
    lr = 0.0001
    run_id = 0
    MODEL_CHECKPOINT = f"../../output/finetune/anomaly/lr={lr}/run_id={run_id}/final-checkpoint"
    INJECTION_METHOD = "IIB+OIB"
    chronos_config = AutoConfig.from_pretrained(MODEL_CHECKPOINT)
    chronos_model_id = "amazon/chronos-t5-small"

    pipeline = ChronosXPipeline(
        prediction_length=prediction_length,
        num_covariates=num_covariates,
        covariate_injection=INJECTION_METHOD,
        pretrained_model_name_or_path=MODEL_CHECKPOINT,
        )
    pipeline.chronosx.eval()

    tokenizer = pipeline.tokenizer

    chronosx_model = pipeline.chronosx


    quantized_test_dataset = ChronosDataset(
        datasets=[test_dataset],
        probabilities=[1.0],
        tokenizer=tokenizer,
        prediction_length=prediction_length,
        min_past=1,
        mode="validation",
    )
    
    
    file_path = "inverse_opt_results_stream.pkl"
    all_batches = load_inverse_opt_results(file_path)

    opt_past_covariates = []
    opt_future_covariates = []
    for elem in all_batches:
        opt_past_covariates.append(elem["past_covariates_final"])
        opt_future_covariates.append(elem["future_covariates_final"])

    all_true = []
    all_medians = []
    all_lows = []
    all_highs = []
    for i, batch in enumerate(quantized_test_dataset):
        if i < 100:
            context = torch.tensor(batch["input_ids"]).unsqueeze(0).to(device).long()
            labels_tokens = torch.tensor(batch["labels"]).unsqueeze(0).to(device).long()

            no_opt_past_covariates = torch.tensor(batch["past_covariates"], device=device).unsqueeze(0).float()
            no_opt_future_covariates = torch.tensor(batch["future_covariates"], device=device).unsqueeze(0).float()
            past_covariates = torch.tensor(opt_past_covariates[i], device=device)#.unsqueeze(0).float()
            future_covariates = torch.tensor(opt_future_covariates[i], device=device)#.unsqueeze(0).float()

            forecasts = chronosx_model.generate(
                input_ids=context,
                max_new_tokens=prediction_length,
                do_sample=True,             
                top_k = 50,
                num_return_sequences=20,
                past_covariates=past_covariates,
                future_covariates=future_covariates
            )

            preds = forecasts[:, 1:].squeeze().cpu().numpy()

            median = np.median(preds)
            low = np.percentile(preds, 10, axis=0)   # 10th percentile → basso
            high = np.percentile(preds, 90, axis=0)  # 90th percentile → alto

            all_medians.append(median)
            all_lows.append(low)
            all_highs.append(high)

            all_true.append(batch["labels"][:1].item())

            #pdb.set_trace()
    
    
    save_dir = "./plots/rolling_inv_opt_forecast.png"
    plot_preds_vs_groundtruth(all_true, all_medians, all_lows, all_highs, save_dir)
    print("Done :)")
    

    """
    all_true = []
    all_medians = []
    all_lows = []
    all_highs = []
    for i, batch in enumerate(quantized_test_dataset):
        if i < 100:
            context = torch.tensor(batch["input_ids"]).unsqueeze(0).to(device).long()
            labels_tokens = torch.tensor(batch["labels"]).unsqueeze(0).to(device).long()

            past_covariates = torch.tensor(batch["past_covariates"], device=device).unsqueeze(0).float()
            future_covariates = torch.tensor(batch["future_covariates"], device=device).unsqueeze(0).float()
            #past_covariates = torch.tensor(opt_past_covariates[i], device=device)#.unsqueeze(0).float()
            #future_covariates = torch.tensor(opt_future_covariates[i], device=device)#.unsqueeze(0).float()

            forecasts = chronosx_model.generate(
                input_ids=context,
                max_new_tokens=prediction_length,
                do_sample=True,             
                top_k = 50,
                num_return_sequences=20,
                past_covariates=past_covariates,
                future_covariates=future_covariates
            )

            preds = forecasts[:, 1:].squeeze().cpu().numpy()

            preds = forecasts[:, 1:].squeeze().cpu().numpy()

            median = np.median(preds)
            low = np.percentile(preds, 10, axis=0)   # 10th percentile → basso
            high = np.percentile(preds, 90, axis=0)  # 90th percentile → alto

            all_medians.append(median)
            all_lows.append(low)
            all_highs.append(high)

            all_true.append(batch["labels"][:1].item())
    
    save_dir = "./plots/rolling_training_preds.png"
    plot_preds_vs_groundtruth(all_true, all_medians, all_lows, all_highs, save_dir)
    """

    #pdb.set_trace()
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

    print(
        {
            #"dataset": dataset_config["name"],
            "dataset": dataset_name,
            "covariate_injection": INJECTION_METHOD,
            **metrics[0],
        }
    )