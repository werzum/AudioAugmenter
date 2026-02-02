from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, List, Sequence
import json
import random
import shutil
import uuid

import numpy as np
import pandas as pd
from pydub import AudioSegment
import soundfile as sf


ALLOWED_EXTENSIONS = {".wav", ".mp3"}


@dataclass
class AudioEntry:
    id: int
    filename: str
    filepath: str
    transcription: str = ""
    augmentations: list[str] = field(default_factory=list)
    is_augmented: bool = False


SESSION_FILENAME = "entries.csv"


class DataManager:
    """Manage audio entries, storage, augmentation, and export."""

    def __init__(self, workspace_dir: Path) -> None:
        self.workspace_dir = Path(workspace_dir)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.session_path = self.workspace_dir / SESSION_FILENAME
        self.entries: list[AudioEntry] = []
        self._next_id = 1

    def load_from_disk(self) -> None:
        """Populate entries from persisted CSV if present."""
        if not self.session_path.exists():
            return
        df = pd.read_csv(self.session_path)
        loaded: list[AudioEntry] = []
        for _, row in df.iterrows():
            try:
                augmentations = json.loads(row.get("augmentations", "[]")) if isinstance(row.get("augmentations"), str) else []
            except json.JSONDecodeError:
                augmentations = []
            is_augmented_raw = row.get("is_augmented", False)
            is_augmented = (
                str(is_augmented_raw).lower() in {"1", "true", "yes"}
                if not isinstance(is_augmented_raw, bool)
                else bool(is_augmented_raw)
            )
            entry = AudioEntry(
                id=int(row["id"]),
                filename=str(row["filename"]),
                filepath=str(row["filepath"]),
                transcription=str(row.get("transcription", "")),
                augmentations=list(augmentations),
                is_augmented=is_augmented,
            )
            if Path(entry.filepath).exists():
                loaded.append(entry)
        self.entries = loaded
        self._next_id = (max((e.id for e in self.entries), default=0) + 1) if self.entries else 1

    def _unique_destination(self, name: str) -> Path:
        dest = self.workspace_dir / name
        stem = dest.stem
        suffix = dest.suffix
        counter = 1
        while dest.exists():
            dest = self.workspace_dir / f"{stem}_{counter}{suffix}"
            counter += 1
        return dest

    def add_files(self, paths: Sequence[Path]) -> list[AudioEntry]:
        added: list[AudioEntry] = []
        for src in paths:
            if not src.exists() or src.suffix.lower() not in ALLOWED_EXTENSIONS:
                continue
            dest = self._unique_destination(src.name)
            shutil.copy2(src, dest)
            entry = AudioEntry(
                id=self._next_id,
                filename=dest.name,
                filepath=str(dest),
            )
            self.entries.append(entry)
            added.append(entry)
            self._next_id += 1
        return added

    def import_dataset(self, dataset_dir: Path) -> list[AudioEntry]:
        dataset_dir = Path(dataset_dir)
        clips_dir = dataset_dir / "clips"
        text_path = dataset_dir / "text.csv"
        if not clips_dir.exists() or not text_path.exists():
            raise ValueError("Expected a folder with clips/ and text.csv.")
        df = pd.read_csv(text_path)
        added: list[AudioEntry] = []
        for _, row in df.iterrows():
            filename = str(row.get("filename", "")).strip()
            if not filename:
                index = str(row.get("index", "")).strip()
                filename = "rec_"+index+".wav"
            src = clips_dir / filename
            if not src.exists() or src.suffix.lower() not in ALLOWED_EXTENSIONS:
                continue
            dest = self._unique_destination(src.name)
            shutil.copy2(src, dest)
            transcription = row.get("transcription", "")
            if pd.isna(transcription):
                transcription = ""
            entry = AudioEntry(
                id=self._next_id,
                filename=dest.name,
                filepath=str(dest),
                transcription=str(transcription),
            )
            self.entries.append(entry)
            added.append(entry)
            self._next_id += 1
        return added

    def list_entries(self) -> list[AudioEntry]:
        return list(self.entries)

    def find_entry(self, entry_id: int) -> AudioEntry | None:
        return next((e for e in self.entries if e.id == entry_id), None)

    def update_transcription(self, entry_id: int, transcription: str) -> None:
        entry = self.find_entry(entry_id)
        if entry:
            entry.transcription = transcription

    def delete_entries(self, entry_ids: Iterable[int], delete_files: bool = False) -> None:
        ids = set(entry_ids)
        remaining: list[AudioEntry] = []
        for entry in self.entries:
            if entry.id in ids:
                if delete_files:
                    path = Path(entry.filepath)
                    if path.exists() and path.is_file():
                        path.unlink()
                continue
            remaining.append(entry)
        self.entries = remaining

    def apply_augmentations(
        self,
        entry_ids: Sequence[int],
        augmentation_names: Sequence[str],
        augmentation_chain: Sequence[object],
    ) -> list[AudioEntry]:
        """Apply selected augmentations sequentially and create one derived entry per source."""
        if not augmentation_names or not augmentation_chain:
            return []
        created: list[AudioEntry] = []
        for entry in self.entries:
            if entry.id not in entry_ids:
                continue
            src_path = Path(entry.filepath)
            if not src_path.exists():
                continue
            samples, sample_rate = sf.read(src_path, always_2d=False)
            samples = samples.astype(np.float32)
            needs_transpose = samples.ndim == 2 and samples.shape[0] > samples.shape[1]
            if needs_transpose:
                samples = samples.T
            augmented = samples
            for augmenter in augmentation_chain:
                augmented = augmenter(samples=augmented, sample_rate=sample_rate)
            if needs_transpose:
                augmented = augmented.T
            combo_name = "_".join(augmentation_names)
            new_name = f"{src_path.stem}_{combo_name}_{uuid.uuid4().hex[:8]}.wav"
            dest = self.workspace_dir / new_name
            sf.write(dest, augmented, sample_rate, format="WAV")
            new_entry = AudioEntry(
                id=self._next_id,
                filename=new_name,
                filepath=str(dest),
                transcription=entry.transcription,
                augmentations=entry.augmentations + list(augmentation_names),
                is_augmented=True,
            )
            self.entries.append(new_entry)
            created.append(new_entry)
            self._next_id += 1
        return created

    def export(
        self,
        entry_ids: Iterable[int],
        output_dir: Path,
        export_format: str = "csv",
        copy_audio: bool = True,
        split_train_test: bool = False,
        train_ratio: float = 0.8,
    ) -> Path:
        """Export selected entries to CSV/JSON or a train/test split."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        selected = sorted([e for e in self.entries if e.id in set(entry_ids)], key=lambda e: e.id)

        def write_text_export(entries: list[AudioEntry], subset_dir: Path) -> None:
            subset_dir.mkdir(parents=True, exist_ok=True)
            filename_map: dict[int, str] = {}
            for idx, entry in enumerate(entries):
                filename_map[entry.id] = f"rec_{idx}.wav"
            df = pd.DataFrame(
                [
                    {
                        "index": idx,
                        "filename": filename_map[e.id],
                        "transcription": e.transcription,
                    }
                    for idx, e in enumerate(entries)
                ]
            )
            export_path = subset_dir / "text.csv"
            df.to_csv(export_path, index=False)
            audio_dir = subset_dir / "clips"
            audio_dir.mkdir(parents=True, exist_ok=True)
            for e in entries:
                src = Path(e.filepath)
                if not src.exists():
                    continue
                target = audio_dir / filename_map[e.id]
                try:
                    audio = AudioSegment.from_file(src)
                    audio.export(target, format="wav")
                except Exception:
                    shutil.copy2(src, target)

        if split_train_test:
            shuffled = list(selected)
            random.shuffle(shuffled)
            total = len(shuffled)
            if total <= 1:
                train_entries = shuffled
                test_entries: list[AudioEntry] = []
            else:
                train_count = int(total * train_ratio)
                train_count = max(1, min(train_count, total - 1))
                train_entries = shuffled[:train_count]
                test_entries = shuffled[train_count:]
            write_text_export(train_entries, output_dir / "train")
            write_text_export(test_entries, output_dir / "test")
            return output_dir

        filename_map: dict[int, str] = {}
        for idx, entry in enumerate(selected):
            filename_map[entry.id] = f"rec_{idx}.wav"
        df = pd.DataFrame(
            [
                {
                    "index": idx,
                    "filename": filename_map[e.id],
                    "transcription": e.transcription,
                }
                for idx, e in enumerate(selected)
            ]
        )
        if export_format.lower() == "json":
            export_path = output_dir / "export.json"
            df.to_json(export_path, orient="records", force_ascii=False, indent=2)
        else:
            export_path = output_dir / "export.csv"
            df.to_csv(export_path, index=False)

        if copy_audio:
            audio_dir = output_dir / "audio"
            audio_dir.mkdir(parents=True, exist_ok=True)
            for e in selected:
                src = Path(e.filepath)
                if not src.exists():
                    continue
                target = audio_dir / filename_map[e.id]
                try:
                    audio = AudioSegment.from_file(src)
                    audio.export(target, format="wav")
                except Exception:
                    # Fallback to raw copy if conversion fails.
                    shutil.copy2(src, target)
        return export_path

    def save_to_disk(self) -> Path:
        """Persist current entries to CSV in the workspace."""
        records = []
        for entry in self.entries:
            data = asdict(entry)
            data["augmentations"] = json.dumps(entry.augmentations)
            records.append(data)
        df = pd.DataFrame(records)
        self.session_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.session_path, index=False)
        return self.session_path
