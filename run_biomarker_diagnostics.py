"""Checkpoint-2 diagnostics for biomarker-head stagnation."""

import torch
from scipy.stats import pearsonr

import run_cell14_eval as cell14


g = cell14.namespace
torch = g["torch"]
np = g["np"]
pd = g["pd"]
json = g["json"]
Path = g["Path"]

DEVICE = g["DEVICE"]
TRAIN_CFG = g["TRAIN_CFG"]
TASK_NAMES = g["TASK_NAMES"]
HemoWaveformDataset = g["HemoWaveformDataset"]
DataLoader = g["DataLoader"]
load_manifest_split = g["load_manifest_split"]
model = g["model"]
checkpoint = g["checkpoint"]
CFG = g["CFG"]
batch_pearson_correlation = g["batch_pearson_correlation"]
frequency_snr_loss = g["frequency_snr_loss"]


def _global_l2_grad_norm(module) -> float:
    """Known-zero-safe L2 norm of all populated parameter gradients."""
    squared_sum = 0.0
    for parameter in module.parameters():
        if parameter.grad is not None:
            squared_sum += float(parameter.grad.detach().float().square().sum().item())
    return squared_sum ** 0.5


def _run_diagnostic_self_test() -> None:
    """Known-answer check for constant-predictor absolute errors."""
    pred = torch.tensor([10.0, 20.0])
    target = torch.tensor([7.0, 25.0])
    assert torch.allclose((pred - target).abs(), torch.tensor([3.0, 5.0]))


_run_diagnostic_self_test()


def biomarker_collapse_guard(preds_arr, targets_arr, masks_arr, task_names, train_means):
    """Assess each real-unit biomarker prediction against a constant baseline."""
    results = {}
    for index, task_name in enumerate(task_names):
        valid = masks_arr[:, index] == 1
        if valid.sum() < 2:
            results[task_name] = {"n": int(valid.sum()), "note": "insufficient samples"}
            continue

        prediction = preds_arr[valid, index]
        target = targets_arr[valid, index]
        mae = float(np.mean(np.abs(prediction - target)))
        trivial_mae = float(np.mean(np.abs(train_means[index] - target)))

        if np.std(prediction) > 1e-6 and np.std(target) > 1e-6:
            r_value = float(pearsonr(target, prediction).statistic)
        else:
            r_value = 0.0

        variance_ratio = float(np.std(prediction) / (np.std(target) + 1e-8))
        results[task_name] = {
            "n": int(valid.sum()),
            "mae": mae,
            "trivial_baseline_mae": trivial_mae,
            "beats_trivial_by_15pct": mae < trivial_mae * 0.85,
            "pearson_r": r_value,
            "variance_ratio": variance_ratio,
            "mode_collapse_likely": (r_value < 0.3) or (variance_ratio < 0.3),
        }
    return results


def _run_collapse_guard_self_test() -> None:
    """Known-answer test: a perfect prediction must not be marked collapsed."""
    result = biomarker_collapse_guard(
        preds_arr=np.asarray([[1.0], [2.0], [3.0]], dtype=np.float32),
        targets_arr=np.asarray([[1.0], [2.0], [3.0]], dtype=np.float32),
        masks_arr=np.ones((3, 1), dtype=np.float32),
        task_names=("known",),
        train_means=np.asarray([2.0], dtype=np.float32),
    )["known"]
    assert result["pearson_r"] > 0.999
    assert result["variance_ratio"] > 0.999
    assert not result["mode_collapse_likely"]


_run_collapse_guard_self_test()

with open(CFG.MANIFEST_PATH, "r", encoding="utf-8") as file:
    manifest_df = pd.DataFrame(json.load(file))

train_manifest, _, test_manifest = load_manifest_split(manifest_df)
train_dataset = HemoWaveformDataset(
    manifest_df=train_manifest,
    db_csv_path=Path(CFG.BASE_DATA_DIR) / "db.csv",
    window_frames=TRAIN_CFG.window_frames,
    random_window=True,
)
test_dataset = g["test_dataset"]

train_loader = DataLoader(
    train_dataset,
    batch_size=TRAIN_CFG.batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=DEVICE.type == "cuda",
)
test_loader = DataLoader(
    test_dataset,
    batch_size=TRAIN_CFG.batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=DEVICE.type == "cuda",
)

mean = checkpoint["biomarker_mean"].float().to(DEVICE)
std = checkpoint["biomarker_std"].float().to(DEVICE)

print("\n" + "=" * 70)
print("CHECKPOINT 2: BIOMARKER-HEAD DIAGNOSTICS")
print("=" * 70)
print("Training-label normalization (mean ± std):")
for index, task_name in enumerate(TASK_NAMES):
    print(f"  {task_name:<22} {mean[index].item():.4f} ± {std[index].item():.4f}")

# The constant baseline predicts the train-only mean that Cell 12 used for
# standardization. It is evaluated on the held-out split with the real mask.
baseline_error_sum = torch.zeros(len(TASK_NAMES), device=DEVICE)
baseline_count = torch.zeros(len(TASK_NAMES), device=DEVICE)

for batch in test_loader:
    target = batch["biomarkers"].to(DEVICE)
    mask = batch["biomarker_mask"].to(DEVICE)
    baseline_error_sum += ((target.nan_to_num() - mean).abs() * mask).sum(dim=0)
    baseline_count += mask.sum(dim=0)

baseline_mae = baseline_error_sum / baseline_count.clamp_min(1.0)
model_mae = torch.tensor(
    [g["test_metrics"]["per_task_mae"][task] for task in TASK_NAMES],
    device=DEVICE,
)

print("\nHeld-out model MAE versus train-mean baseline:")
for index, task_name in enumerate(TASK_NAMES):
    improvement = baseline_mae[index] - model_mae[index]
    print(
        f"  {task_name:<22} model={model_mae[index].item():.4f} | "
        f"mean-baseline={baseline_mae[index].item():.4f} | "
        f"improvement={improvement.item():+.4f}"
    )

# Collect full held-out predictions in real units for the collapse guard.
model.eval()
prediction_rows, target_rows, mask_rows = [], [], []
with torch.no_grad():
    for batch in test_loader:
        x = batch["x"].to(DEVICE)
        with torch.cuda.amp.autocast(enabled=DEVICE.type == "cuda"):
            output = model(x)
        prediction_rows.append((output["biomarkers"].float() * std + mean).cpu().numpy())
        target_rows.append(batch["biomarkers"].cpu().numpy())
        mask_rows.append(batch["biomarker_mask"].cpu().numpy())

guard_results = biomarker_collapse_guard(
    preds_arr=np.concatenate(prediction_rows, axis=0),
    targets_arr=np.concatenate(target_rows, axis=0),
    masks_arr=np.concatenate(mask_rows, axis=0),
    task_names=TASK_NAMES,
    train_means=mean.detach().cpu().numpy(),
)

print("\nHeld-out collapse guard:")
for task_name in TASK_NAMES:
    result = guard_results[task_name]
    print(
        f"  {task_name:<22} r={result['pearson_r']:+.4f} | "
        f"var_ratio={result['variance_ratio']:.4f} | "
        f"beats_mean_15%={result['beats_trivial_by_15pct']} | "
        f"collapse_likely={result['mode_collapse_likely']}"
    )

# Compare gradient signal entering the shared trunk from waveform and biomarker
# objectives using one real training batch. Both losses are in float32.
batch = next(iter(train_loader))
x = batch["x"].to(DEVICE)
bvp_target = batch["bvp_target"].to(DEVICE)
bio_raw = batch["biomarkers"].to(DEVICE)
bio_mask = batch["biomarker_mask"].to(DEVICE)
fps = batch["video_fps"].to(DEVICE)
bio_target = torch.where(
    bio_mask.bool(),
    (bio_raw - mean) / std,
    torch.zeros_like(bio_raw),
)

model.train()

model.zero_grad(set_to_none=True)
prediction = model(x)
waveform_loss = (
    (1.0 - batch_pearson_correlation(prediction["bvp"], bvp_target)).mean()
    + TRAIN_CFG.lambda_snr * frequency_snr_loss(prediction["bvp"], bvp_target, fps)
)
waveform_loss.backward()
waveform_trunk_grad = _global_l2_grad_norm(model.input_projection)
waveform_head_grad = _global_l2_grad_norm(model.waveform_head)
waveform_bio_head_grad = _global_l2_grad_norm(model.biomarker_head)

model.zero_grad(set_to_none=True)
prediction = model(x)
valid_per_task = bio_mask.sum(dim=0)
task_mse = ((prediction["biomarkers"].float() - bio_target.float()).square() * bio_mask).sum(dim=0)
task_mse = task_mse / valid_per_task.clamp_min(1.0)
biomarker_loss = task_mse[valid_per_task > 0].mean()
biomarker_loss.backward()
biomarker_trunk_grad = _global_l2_grad_norm(model.input_projection)
biomarker_head_grad = _global_l2_grad_norm(model.biomarker_head)
biomarker_wave_head_grad = _global_l2_grad_norm(model.waveform_head)
model.zero_grad(set_to_none=True)
model.eval()

print("\nOne-real-batch gradient norms:")
print(f"  waveform loss: {waveform_loss.item():.6f}")
print(f"  biomarker loss: {biomarker_loss.item():.6f}")
print(f"  shared trunk from waveform loss: {waveform_trunk_grad:.6e}")
print(f"  shared trunk from biomarker loss: {biomarker_trunk_grad:.6e}")
print(f"  waveform head from waveform loss: {waveform_head_grad:.6e}")
print(f"  biomarker head from biomarker loss: {biomarker_head_grad:.6e}")
print(f"  biomarker head from waveform loss: {waveform_bio_head_grad:.6e}")
print(f"  waveform head from biomarker loss: {biomarker_wave_head_grad:.6e}")
