# AudioAugmenter

Desktop GUI for importing, augmenting, and exporting WAV/MP3 samples.

## Features
- Drag-and-drop import into a local `workspace/` directory.
- Central table to edit transcriptions, delete rows, and track applied augmentations.
- One-click augmentation (Gaussian noise, time stretch, pitch shift, shift) via `audiomentations`.
- Export to CSV or JSON with optional audio copies.

## Requirements
- Python 3.13+
- Dependencies: PyQt6, audiomentations, pandas, numpy, soundfile, pydub (see `pyproject.toml`).

## Quick start
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
python app.py
```

## Usage
- Use **Add Files** or drag `.wav`/`.mp3` files into the left pane. Files are copied into `workspace/`.
- Edit transcriptions directly in the middle table.
- Select rows, check one or more augmentations (edit parameters in the panel), and click **Apply Augmentation** to create a single derived file per source with all checked augmentations applied in order.
- Delete selected rows and their workspace files with **Delete Selected**.
- Export selected rows (or all if none selected) to CSV/JSON; exported rows are reindexed from 0 and filenames reset to `rec_<n>.wav`; audio copies go to `exports/audio/` by default.
- Use **Save to workspace** to persist the current table to `workspace/entries.csv`; restarting the app reloads this state.
- Double-click a row to play the audio (requires pydub playback support, e.g., simpleaudio).

## Notes
- Augmentations rely on `audiomentations` and `soundfile`; ensure their native deps are available on your system.
- Augmented outputs are written as WAV files to maximize compatibility.
- Playback is not included; integrate `pydub`/`sounddevice` if needed.
