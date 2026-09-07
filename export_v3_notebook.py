"""Mechanical export of the V3 source cells plus the feasibility audit to ipynb."""
import json
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "HemoVision_V3_Pipeline.py"
AUDIT = ROOT / "cell16_hemoglobin_feasibility_audit.py"
OUTPUT = ROOT / "HemoVision_V3_Pipeline.ipynb"


def source_lines(text: str):
    return [line + "\n" for line in text.splitlines()]


def make_cell(chunk: str):
    lines = chunk.strip("\n").splitlines()
    if lines and lines[0].strip() == "[markdown]":
        markdown = []
        for line in lines[1:]:
            markdown.append((line[2:] if line.startswith("# ") else line) + "\n")
        return {"cell_type": "markdown", "metadata": {}, "source": markdown}
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source_lines(chunk.strip("\n"))}


text = SOURCE.read_text(encoding="utf-8")
chunks = text.split("# %%")
cells = [make_cell(chunk) for chunk in chunks if chunk.strip()]
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# Cell 7: Hemoglobin feasibility and confound audit\n",
        "\n",
        "Runs train-only non-video baselines and records the audit result before interpreting any video-model hemoglobin score.\n",
    ],
})
cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source_lines(AUDIT.read_text(encoding="utf-8"))})
for cell in cells:
    cell["id"] = uuid.uuid4().hex[:12]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
OUTPUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
assert len(cells) == 9, f"Unexpected notebook cell count: {len(cells)}"
print(f"Wrote {OUTPUT} with {len(cells)} cells.")
