import torch
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path
from v3.config import NUM_ROIS, CHANNELS_PER_ROI, TOTAL_CHANNELS, TASK_COLUMNS, TASK_NAMES, TRAIN_CFG, CAMERA_TYPES
from v3.dataset import HemoWaveformDataset
from v3.model import HemoVisionBoundedTCN_V3
from v3.loss import HemoMultiTaskLoss_V3

def test_pipeline():
    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        cache_path = root / "7_FullHDwebcam_before.pt"
        sync_path = root / "7_FullHDwebcam_before.txt"
        db_path = root / "db.csv"

        torch.save({"signal": torch.arange(TOTAL_CHANNELS * TRAIN_CFG.window_frames, dtype=torch.float32).reshape(NUM_ROIS, CHANNELS_PER_ROI, TRAIN_CFG.window_frames)}, cache_path)
        np.savetxt(sync_path, np.column_stack((np.arange(TRAIN_CFG.window_frames, dtype=np.float32), np.arange(TRAIN_CFG.window_frames, dtype=np.float32) / 30.0)))

        db_row = {"patient_id": 7, "video": "video/7_FullHDwebcam_before.avi", "ppg_sync": str(sync_path)}
        db_row.update({column: float(index + 1) for index, column in enumerate(TASK_COLUMNS)})
        pd.DataFrame([db_row]).to_csv(db_path, index=False)

        manifest = pd.DataFrame([{"roi_cache_path": str(cache_path), "subject_id": "7", "split": "train", "num_frames": TRAIN_CFG.window_frames, "video_fps": 30.0}])
        dataset = HemoWaveformDataset(manifest_df=manifest, db_csv_path=db_path, window_frames=TRAIN_CFG.window_frames, random_window=False)
        sample = dataset[0]

        print("Dataset Shapes:")
        for k, v in sample.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: {v.shape}")
            else:
                print(f"  {k}: {type(v)}")
                
        assert sample["x"].shape == (TOTAL_CHANNELS, TRAIN_CFG.window_frames)
        assert sample["x_raw"].shape == (TOTAL_CHANNELS, TRAIN_CFG.window_frames)
        assert sample["bvp_target"].shape == (1, TRAIN_CFG.window_frames)
        assert sample["biomarkers"].shape == (len(TASK_NAMES),)
        assert sample["camera_onehot"].shape == (len(CAMERA_TYPES),)

        print("\nModel Forward Pass:")
        model = HemoVisionBoundedTCN_V3()
        # Create a dummy batch of size 2
        x = torch.stack([sample["x"], sample["x"]])
        x_raw = torch.stack([sample["x_raw"], sample["x_raw"]])
        cam = torch.stack([sample["camera_onehot"], sample["camera_onehot"]])
        
        preds = model(x, x_raw, cam)
        print(f"  bvp pred: {preds['bvp'].shape}")
        print(f"  biomarkers pred: {preds['biomarkers'].shape}")
        
        assert preds["bvp"].shape == (2, 1, TRAIN_CFG.window_frames)
        assert preds["biomarkers"].shape == (2, len(TASK_NAMES))

        print("\nLoss Calculation:")
        criterion = HemoMultiTaskLoss_V3()
        bvp_t = torch.stack([sample["bvp_target"], sample["bvp_target"]])
        bio_t = torch.stack([sample["biomarkers"], sample["biomarkers"]])
        mask_t = torch.stack([sample["biomarker_mask"], sample["biomarker_mask"]])
        fps_t = torch.stack([sample["video_fps"], sample["video_fps"]])
        
        losses = criterion(preds, bvp_t, bio_t, mask_t, fps_t)
        print(f"  Total Loss: {losses['total'].item():.4f}")
        print(f"  Biomarker Loss: {losses['biomarker'].item():.4f}")
        
        print("\nSMOKE TEST PASSED!")

if __name__ == "__main__":
    test_pipeline()
