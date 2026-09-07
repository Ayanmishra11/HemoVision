"""Execute the fixed Cell 14 in a clean Python process for verification."""

import logging
from pathlib import Path

import torch


SOURCE_PATH = Path(r"C:\Users\VICTUS\Downloads\new.py")
CELL14_PATH = Path(__file__).with_name("heldout_test_cell14.py")


def section(source: str, start_marker: str, end_marker: str) -> str:
    start = source.index(start_marker)
    end = source.index(end_marker, start)
    return source[start:end]


source = SOURCE_PATH.read_text(encoding="utf-8")
namespace = {"__name__": "__cell14_verification__"}
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
namespace["log"] = logging.getLogger("HemoVisionV2")

# Reuse the exact Cell 2, Cell 10, and Cell 11 definitions that produced the
# checkpoint. This intentionally does not execute extraction or training cells.
exec(section(source, "# CELL 2: CENTRAL CONFIGURATION", "# CELL 3:"), namespace)
exec(section(source, "# CELL 10: ALIGNED WAVEFORM + BIOMARKER DATASET", "# CELL 11:"), namespace)
exec(section(source, "# CELL 11: BOUNDED TCN + MULTI-TASK LOSS", "# CELL 12:"), namespace)

split_start = source.index("def load_manifest_split(")
split_end = source.index("def fft_hr_bpm(", split_start)
exec(source[split_start:split_end], namespace)

fft_start = source.index("def fft_hr_bpm(")
fft_end = source.index("def build_loader(", fft_start)
exec(source[fft_start:fft_end], namespace)
namespace["DEVICE"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")

exec(CELL14_PATH.read_text(encoding="utf-8"), namespace)
