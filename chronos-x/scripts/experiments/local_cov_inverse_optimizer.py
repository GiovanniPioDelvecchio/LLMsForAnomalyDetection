import torch
import pandas as pd
from pathlib import Path
from datasets import Dataset

from transformers import AutoConfig

from chronosx.chronosx import ChronosXPipeline
from chronosx.utils.chronos_dataset import ChronosDataset
#from chronosx.utils.hf_data_loader import to_gluonts_univariate
#from inv_utils import add_dates_and_convert_to_gluonts


import pickle
from pathlib import Path

from inv_utils import load_anomaly_dataset

import pdb

# -------------------- Main --------------------

print("@@ START OF THE SCRIPT @@")
dataset_name = "anomaly"
prediction_length = 1
num_covariates = 2

train_dataset, val_dataset, test_dataset = load_anomaly_dataset()

print("@@ LOADING MODEL @@")
lr = 0.0001
run_id = 1
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

print("@@ RUNNING MANUAL EVALUATION @@")
chronosx_model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
chronosx_model.to(device)

save_path = Path("inverse_opt_results_stream.pkl")
if save_path.exists():
    print(f"⚠️ Rimuovo file precedente: {save_path}")
    save_path.unlink()

num_epochs = 1000  # puoi ridurre se vedi che ci mette troppo

for i, batch in enumerate(quantized_test_dataset):
    context = torch.tensor(batch["input_ids"]).unsqueeze(0).to(device).long()
    labels_tokens = torch.tensor(batch["labels"]).unsqueeze(0).to(device).long()

    past_covariates = torch.tensor(batch["past_covariates"], device=device).unsqueeze(0).float()
    future_covariates = torch.tensor(batch["future_covariates"], device=device).unsqueeze(0).float()

    # initial_covariates = past_covariates.clone().detach() # this was created to verify the legality of negative values 
                                                            # obtained through inverse optimization

    past_covariates.requires_grad_(True)
    future_covariates.requires_grad_(True)

    # ---- Costruzione maschere ----
    # escludi padding (==0.0) e tieni tutto il resto
    past_mask = (past_covariates != 0.0).float()
    future_mask = (future_covariates != 0.0).float()

    # se invece vuoi ottimizzare solo i valori == 0.5:
    # past_mask = (past_covariates == 0.5).float()
    # future_mask = (future_covariates == 0.5).float()

    # ---- Hook per azzerare gradienti fuori mask ----
    def apply_mask(grad, mask):
        return grad * mask

    past_covariates.register_hook(lambda grad: apply_mask(grad, past_mask))
    future_covariates.register_hook(lambda grad: apply_mask(grad, future_mask))

    optimizer = torch.optim.Adam([past_covariates, future_covariates], lr=1e-3)

    #pdb.set_trace()

    print(f"\n[Batch {i}]")
    print(f" context: {context.shape}")
    print(f" past_covariates: {past_covariates.shape}")
    print(f" future_covariates: {future_covariates.shape}")
    print(f" labels_tokens: {labels_tokens.shape}")

    batch_preds = []
    batch_losses = []

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        decoder_input = torch.tensor([[0, 0]], device=context.device)  

        outputs = chronosx_model(
            input_ids=context,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            decoder_input_ids=decoder_input,
        )

        logits = outputs.logits[:, -1, :]   # [1, vocab]
        labels_aligned = labels_tokens[:, 0]   # primo token GT

        loss = torch.nn.CrossEntropyLoss()(logits, labels_aligned)
        loss.backward()
        optimizer.step()

        pred_token = logits.argmax(dim=-1)

        with torch.no_grad():
            probs = torch.softmax(logits, -1)
            prob_correct = probs[0, labels_aligned].item()
            cov_stats = {
                "past_min": past_covariates.min().item(),
                "past_max": past_covariates.max().item(),
                "future_min": future_covariates.min().item(),
                "future_max": future_covariates.max().item(),
            }
        
        if epoch % 50 == 0:  # log ogni 50 step
            print(f"[Epoch {epoch}] Loss={loss.item():.4f}, Prob_correct={prob_correct:.3f}, Cov_stats={cov_stats}")
            #pdb.set_trace()

        batch_preds.append(pred_token.item())
        batch_losses.append(loss.item())

        if loss.item() < 0.1:  # early stopping aggressivo opzionale
            print(f"Early stopping at epoch {epoch}, loss={loss.item():.4f}")
            break

    # salva risultati di questo batch in append
    batch_result = {
        "batch_idx": i,
        "gt": labels_tokens[:, 0].item(),
        "preds": batch_preds,
        "losses": batch_losses,
        "past_covariates_final": past_covariates.detach().cpu().numpy(),
        "future_covariates_final": future_covariates.detach().cpu().numpy(),
    }

    with open(save_path, "ab") as f:
        pickle.dump(batch_result, f)
        f.flush()

    print(f"✅ Batch {i} salvato in {save_path}")

print(f"@@ FINISHED, risultati salvati progressivamente in {save_path} @@")
