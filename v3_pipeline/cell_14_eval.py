# %%
# ==========================================
# CELL 14: BIOMARKER COLLAPSE DIAGNOSTIC (masked)
# ==========================================
import numpy as np
from scipy.stats import pearsonr
import torch

CHECKPOINT_PATH_DIAG = DATASET_ROOT / TRAIN_CFG.checkpoint_name
checkpoint = torch.load(CHECKPOINT_PATH_DIAG, map_location=DEVICE)

eval_model = HemoVisionBoundedTCN().to(DEVICE)
eval_model.load_state_dict(checkpoint["model_state_dict"])
eval_model.eval()

ckpt_biomarker_mean = checkpoint["biomarker_mean"].to(DEVICE)
ckpt_biomarker_std = checkpoint["biomarker_std"].to(DEVICE)

all_preds, all_targets, all_masks = [], [], []

print(f"Running Diagnostic on Checkpoint Epoch: {checkpoint['epoch']}")

with torch.no_grad():
    for batch in test_loader:
        x = batch['x'].to(DEVICE, non_blocking=True)
        x_raw = batch['x_raw'].to(DEVICE, non_blocking=True) # EXTRACT x_raw
        targets = batch['biomarkers'].cpu().numpy()
        masks = batch['biomarker_mask'].cpu().numpy()

        with torch.cuda.amp.autocast(enabled=USE_AMP):
            outputs = eval_model(x, x_raw) # PASS x_raw

        preds_norm = outputs['biomarkers'].cpu().numpy()
        mean_np = ckpt_biomarker_mean.cpu().numpy()
        std_np = ckpt_biomarker_std.cpu().numpy()
        preds_orig = (preds_norm * std_np) + mean_np

        all_preds.append(preds_orig)
        all_targets.append(targets)
        all_masks.append(masks)

all_preds = np.concatenate(all_preds, axis=0)
all_targets = np.concatenate(all_targets, axis=0)
all_masks = np.concatenate(all_masks, axis=0)

print(f"\n{'Target':<20} | {'n valid':<8} | {'Pearson r':<12} | {'Pred Var / Target Var':<22} | {'Beats mean?':<12} | {'Status'}")
print("-" * 100)

for i, task_name in enumerate(TASK_NAMES):
    valid = all_masks[:, i] == 1
    if valid.sum() < 2:
        print(f"{task_name:<20} | {int(valid.sum()):<8} | insufficient samples")
        continue

    p = all_preds[valid, i]
    t = all_targets[valid, i]

    r_val, _ = pearsonr(p, t) if np.std(p) > 1e-6 and np.std(t) > 1e-6 else (0.0, 1.0)
    var_ratio = np.var(p) / (np.var(t) + 1e-8)

    trivial_mae = float(np.mean(np.abs(ckpt_biomarker_mean.cpu().numpy()[i] - t)))
    real_mae = float(np.mean(np.abs(p - t)))
    beats_mean = real_mae < trivial_mae * 0.85

    status = "OK" if (r_val > 0.15 and var_ratio > 0.10 and beats_mean) else "COLLAPSED"

    print(f"{task_name:<20} | {int(valid.sum()):<8} | {r_val:<+12.4f} | {var_ratio:<22.4f} | {str(beats_mean):<12} | {status}")