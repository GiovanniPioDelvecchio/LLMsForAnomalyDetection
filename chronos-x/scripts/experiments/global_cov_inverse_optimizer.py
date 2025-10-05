import torch
from pathlib import Path
import pickle

from transformers import AutoConfig
from chronosx.chronosx import ChronosXPipeline
from chronosx.utils.chronos_dataset import ChronosDataset
from inv_utils import load_anomaly_dataset

import pdb
# -------------------- Main --------------------

print("@@ START OF GLOBAL OPTIMIZATION SCRIPT @@")

dataset_name = "anomaly"
prediction_length = 1
num_covariates = 2

# carica dataset
train_dataset, test_dataset, rolling_test_dataset = load_anomaly_dataset()

# carica modello
MODEL_CHECKPOINT = "../../output/finetune/anomaly/lr=0.001/run_id=2/final-checkpoint"
INJECTION_METHOD = "IIB+OIB"
chronos_config = AutoConfig.from_pretrained(MODEL_CHECKPOINT)

pipeline = ChronosXPipeline(
    prediction_length=prediction_length,
    num_covariates=num_covariates,
    covariate_injection=INJECTION_METHOD,
    pretrained_model_name_or_path=MODEL_CHECKPOINT,
)

tokenizer = pipeline.tokenizer
chronosx_model = pipeline.chronosx
chronosx_model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
chronosx_model.to(device)


# dataset quantizzato
quantized_val_dataset = ChronosDataset(
    datasets=[rolling_test_dataset],
    probabilities=[1.0],
    tokenizer=tokenizer,
    prediction_length=prediction_length,
    min_past=1,
    mode="validation",
)

pdb.set_trace()

# -------------------- GLOBAL COVARIATES --------------------
# Costruisci covariates globali da tutto il dataset
# Qui semplifico: assumo che rolling_test_dataset contenga un'unica serie

# Perché non si può fare così:
# 1. Ricostruzione → slicing → ricostruzione
# 2. Disconnessione del grafo dei gradienti
# 3. Duplicazione e conflitto degli aggiornamenti
# 4. Computazionalmente inefficiente
raw_covariates = torch.tensor(rolling_test_dataset[0]["feat_dynamic_real"]).T  # shape (T, F)
T, F = raw_covariates.shape
print(f"Global covariates shape: {raw_covariates.shape}")

# rendilo parametro ottimizzabile
global_covariates = torch.nn.Parameter(raw_covariates.clone().float().to(device))

# ottimizzatore
optimizer = torch.optim.Adam([global_covariates], lr=1e-3)

# -------------------- LOOP DI TRAINING --------------------
num_epochs = 50
save_path = Path("global_inverse_opt.pkl")
if save_path.exists():
    save_path.unlink()

for epoch in range(num_epochs):
    total_loss = 0.0
    optimizer.zero_grad()

    # accumula loss su tutti i batch
    for i, batch in enumerate(quantized_val_dataset):
        context = torch.tensor(batch["input_ids"]).unsqueeze(0).to(device).long()
        labels_tokens = torch.tensor(batch["labels"]).unsqueeze(0).to(device).long()

        # slice da global_covariates
        # (batch["past_indices"], batch["future_indices"] ipotetici — dipende da come ChronosDataset fornisce gli indici)
        past_idx = batch["past_indices"]
        future_idx = batch["future_indices"]

        past_covariates = global_covariates[past_idx].unsqueeze(0)   # (1, L_past, F)
        future_covariates = global_covariates[future_idx].unsqueeze(0) # (1, L_future, F)

        decoder_input = torch.tensor([[0, 0]], device=device)

        outputs = chronosx_model(
            input_ids=context,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            decoder_input_ids=decoder_input,
        )

        logits = outputs.logits[:, -1, :]   # (1, vocab)
        labels_aligned = labels_tokens[:, 0]

        loss = torch.nn.CrossEntropyLoss()(logits, labels_aligned)
        total_loss += loss

    # backward su loss media
    total_loss = total_loss / len(quantized_val_dataset)
    total_loss.backward()
    optimizer.step()

    print(f"[Epoch {epoch}] Avg Loss = {total_loss.item():.4f}")

    # salva checkpoint progressivo
    with open(save_path, "wb") as f:
        pickle.dump(global_covariates.detach().cpu().numpy(), f)

print(f"@@ FINISHED, global covariates salvate in {save_path} @@")
