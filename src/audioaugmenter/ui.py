from __future__ import annotations

from pathlib import Path
from typing import Iterable, List
import tempfile
import threading

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets

from .augmentations import DEFAULT_PARAMS, build_augmenter, list_augmentation_names
from .data_manager import ALLOWED_EXTENSIONS, AudioEntry, DataManager
from pydub import AudioSegment
from pydub.playback import play
import soundfile as sf


class DropListWidget(QtWidgets.QListWidget):
    filesDropped = QtCore.pyqtSignal(list)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event: QtGui.QDragMoveEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event: QtGui.QDropEvent) -> None:
        if event.mimeData().hasUrls():
            paths = [Path(url.toLocalFile()) for url in event.mimeData().urls()]
            self.filesDropped.emit(paths)
            event.acceptProposedAction()
        else:
            super().dropEvent(event)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Audio Augmenter")
        self.resize(1200, 700)
        self.data_manager = DataManager(workspace_dir=Path.cwd() / "workspace")
        self.augmentation_names = list_augmentation_names()
        self.augmentation_param_widgets: dict[str, dict[str, QtWidgets.QDoubleSpinBox]] = {}
        self.params_stack: QtWidgets.QStackedWidget | None = None
        self._suppress_table_signals = False

        self._build_ui()
        self._connect_signals()
        self._load_session()

    # UI construction helpers
    def _build_ui(self) -> None:
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_middle_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setStretchFactor(1, 2)
        splitter.setStretchFactor(2, 1)
        self.setCentralWidget(splitter)
        self.statusBar().showMessage("Ready")

    def _build_left_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        label = QtWidgets.QLabel("Drop WAV/MP3 files here or use Add Files")
        label.setWordWrap(True)
        self.source_list = DropListWidget()
        self.add_files_button = QtWidgets.QPushButton("Add Files")
        self.import_dataset_button = QtWidgets.QPushButton("Import Dataset")

        layout.addWidget(label)
        layout.addWidget(self.source_list, stretch=1)
        layout.addWidget(self.add_files_button)
        layout.addWidget(self.import_dataset_button)
        return panel

    def _build_middle_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)

        self.table = QtWidgets.QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(
            ["File Name", "File Path", "Transcription", "Augmentation Applied"]
        )
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.verticalHeader().setVisible(False)

        controls = QtWidgets.QHBoxLayout()
        self.augmentation_list = QtWidgets.QListWidget()
        self.augmentation_list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        for name in self.augmentation_names:
            item = QtWidgets.QListWidgetItem(name)
            item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.CheckState.Unchecked)
            self.augmentation_list.addItem(item)
        if self.augmentation_list.count():
            self.augmentation_list.setCurrentRow(0)

        self.params_stack = QtWidgets.QStackedWidget()
        self._build_param_forms()

        self.apply_aug_button = QtWidgets.QPushButton("Apply Augmentation")
        self.preview_aug_button = QtWidgets.QPushButton("Preview Augmentation")
        self.delete_button = QtWidgets.QPushButton("Delete Selected")
        self.save_button = QtWidgets.QPushButton("Save to workspace")

        left_box = QtWidgets.QVBoxLayout()
        left_box.addWidget(QtWidgets.QLabel("Augmentations:"))
        left_box.addWidget(self.augmentation_list)

        right_box = QtWidgets.QVBoxLayout()
        right_box.addWidget(QtWidgets.QLabel("Parameters"))
        right_box.addWidget(self.params_stack)

        controls.addLayout(left_box)
        controls.addLayout(right_box)
        controls.addWidget(self.apply_aug_button)
        controls.addWidget(self.preview_aug_button)
        controls.addStretch()
        controls.addWidget(self.save_button)
        controls.addWidget(self.delete_button)

        layout.addLayout(controls)
        layout.addWidget(self.table, stretch=1)
        return panel

    def _build_right_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QGroupBox("Export")
        layout = QtWidgets.QVBoxLayout(panel)

        self.export_selection_list = QtWidgets.QListWidget()
        self.export_format_combo = QtWidgets.QComboBox()
        self.export_format_combo.addItems(["csv", "json"])
        self.copy_audio_checkbox = QtWidgets.QCheckBox("Copy audio files")
        self.copy_audio_checkbox.setChecked(True)
        self.split_export_checkbox = QtWidgets.QCheckBox("Export train/test split")
        self.train_ratio_spin = QtWidgets.QDoubleSpinBox()
        self.train_ratio_spin.setRange(50.0, 95.0)
        self.train_ratio_spin.setSingleStep(5.0)
        self.train_ratio_spin.setSuffix("% train")
        self.train_ratio_spin.setValue(80.0)
        self.train_ratio_spin.setEnabled(False)

        path_layout = QtWidgets.QHBoxLayout()
        self.output_path_edit = QtWidgets.QLineEdit(str(Path.cwd() / "exports"))
        self.browse_button = QtWidgets.QPushButton("Browse")
        path_layout.addWidget(self.output_path_edit)
        path_layout.addWidget(self.browse_button)

        self.export_button = QtWidgets.QPushButton("Export")

        layout.addWidget(QtWidgets.QLabel("Selected for export:"))
        layout.addWidget(self.export_selection_list, stretch=1)
        layout.addWidget(QtWidgets.QLabel("Format"))
        layout.addWidget(self.export_format_combo)
        layout.addWidget(self.copy_audio_checkbox)
        layout.addWidget(self.split_export_checkbox)
        layout.addWidget(QtWidgets.QLabel("Train ratio"))
        layout.addWidget(self.train_ratio_spin)
        layout.addLayout(path_layout)
        layout.addWidget(self.export_button)
        return panel

    def _connect_signals(self) -> None:
        self.source_list.filesDropped.connect(self._handle_dropped_files)
        self.add_files_button.clicked.connect(self._open_file_dialog)
        self.import_dataset_button.clicked.connect(self._import_dataset)
        self.apply_aug_button.clicked.connect(self._apply_augmentation)
        self.preview_aug_button.clicked.connect(self._preview_augmentation)
        self.delete_button.clicked.connect(self._delete_selected)
        self.save_button.clicked.connect(self._save_session)
        self.export_button.clicked.connect(self._export_selected)
        self.browse_button.clicked.connect(self._browse_export_dir)
        self.table.itemChanged.connect(self._handle_item_changed)
        self.table.cellDoubleClicked.connect(self._handle_row_double_clicked)
        self.table.selectionModel().selectionChanged.connect(self._update_export_selection)
        self.augmentation_list.itemClicked.connect(self._handle_param_focus)
        self.split_export_checkbox.toggled.connect(self._toggle_split_export_options)

    def _load_session(self) -> None:
        """Load persisted entries from workspace CSV."""
        self.data_manager.load_from_disk()
        self._refresh_table()

    def _build_param_forms(self) -> None:
        """Create parameter editors for each augmentation."""
        assert self.params_stack is not None
        for name in self.augmentation_names:
            form = QtWidgets.QWidget()
            layout = QtWidgets.QFormLayout(form)
            widgets: dict[str, QtWidgets.QDoubleSpinBox] = {}
            defaults = DEFAULT_PARAMS.get(name, {})

            def make_spin(minimum: float, maximum: float, step: float, value: float) -> QtWidgets.QDoubleSpinBox:
                spin = QtWidgets.QDoubleSpinBox()
                spin.setRange(minimum, maximum)
                spin.setSingleStep(step)
                spin.setValue(value)
                return spin

            if name == "gaussian_noise":
                widgets["min_amplitude"] = make_spin(0.0, 0.2, 0.001, defaults["min_amplitude"])
                widgets["max_amplitude"] = make_spin(0.0, 0.5, 0.001, defaults["max_amplitude"])
                layout.addRow("Min amplitude", widgets["min_amplitude"])
                layout.addRow("Max amplitude", widgets["max_amplitude"])
            elif name == "time_stretch":
                widgets["min_rate"] = make_spin(0.5, 2.0, 0.01, defaults["min_rate"])
                widgets["max_rate"] = make_spin(0.5, 3.0, 0.01, defaults["max_rate"])
                layout.addRow("Min rate", widgets["min_rate"])
                layout.addRow("Max rate", widgets["max_rate"])
            elif name == "pitch_shift":
                widgets["min_semitones"] = make_spin(-12.0, 0.0, 0.5, defaults["min_semitones"])
                widgets["max_semitones"] = make_spin(0.0, 12.0, 0.5, defaults["max_semitones"])
                layout.addRow("Min semitones", widgets["min_semitones"])
                layout.addRow("Max semitones", widgets["max_semitones"])

            self.augmentation_param_widgets[name] = widgets
            self.params_stack.addWidget(form)
        if self.params_stack.count():
            self.params_stack.setCurrentIndex(0)

    # Event handlers
    def _open_file_dialog(self) -> None:
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Select audio files",
            str(Path.home()),
            "Audio Files (*.wav *.mp3)",
        )
        if paths:
            self._add_files([Path(p) for p in paths])

    def _handle_dropped_files(self, paths: List[Path]) -> None:
        self._add_files(paths)

    def _import_dataset(self) -> None:
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select dataset folder",
            str(Path.cwd()),
        )
        if not directory:
            return
        try:
            added = self.data_manager.import_dataset(Path(directory))
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "Import failed", str(exc))
            return
        if not added:
            QtWidgets.QMessageBox.information(
                self,
                "Nothing imported",
                "No valid clips found in the selected dataset.",
            )
            return
        self._refresh_table()
        self.statusBar().showMessage(f"Imported {len(added)} clip(s)", 3000)

    def _add_files(self, paths: Iterable[Path]) -> None:
        added = self.data_manager.add_files(list(paths))
        if not added:
            QtWidgets.QMessageBox.information(self, "No files added", "No valid WAV/MP3 files detected.")
            return
        self._refresh_table()
        self.source_list.clear()
        self.statusBar().showMessage(f"Added {len(added)} file(s)", 3000)

    def _refresh_table(self) -> None:
        entries = self.data_manager.list_entries()
        self._suppress_table_signals = True
        self.table.setRowCount(len(entries))
        for row, entry in enumerate(entries):
            self._set_table_row(row, entry)
        self._suppress_table_signals = False
        self._update_export_selection()

    def _set_table_row(self, row: int, entry: AudioEntry) -> None:
        name_item = QtWidgets.QTableWidgetItem(entry.filename)
        name_item.setData(QtCore.Qt.ItemDataRole.UserRole, entry.id)
        name_item.setFlags(name_item.flags() ^ QtCore.Qt.ItemFlag.ItemIsEditable)

        path_item = QtWidgets.QTableWidgetItem(entry.filepath)
        path_item.setToolTip(entry.filepath)
        path_item.setFlags(path_item.flags() ^ QtCore.Qt.ItemFlag.ItemIsEditable)

        transcription_item = QtWidgets.QTableWidgetItem(entry.transcription)
        augmentation_item = QtWidgets.QTableWidgetItem(", ".join(entry.augmentations) if entry.augmentations else "None")
        augmentation_item.setFlags(augmentation_item.flags() ^ QtCore.Qt.ItemFlag.ItemIsEditable)

        self.table.setItem(row, 0, name_item)
        self.table.setItem(row, 1, path_item)
        self.table.setItem(row, 2, transcription_item)
        self.table.setItem(row, 3, augmentation_item)

    def _selected_entry_ids(self) -> list[int]:
        ids: list[int] = []
        for index in self.table.selectionModel().selectedRows():
            item = self.table.item(index.row(), 0)
            if item:
                entry_id = item.data(QtCore.Qt.ItemDataRole.UserRole)
                if isinstance(entry_id, int):
                    ids.append(entry_id)
        return ids

    def _apply_augmentation(self) -> None:
        entry_ids = self._selected_entry_ids()
        if not entry_ids:
            QtWidgets.QMessageBox.information(self, "No selection", "Select one or more rows to augment.")
            return
        aug_names = self._selected_augmentations()
        if not aug_names:
            QtWidgets.QMessageBox.information(self, "No augmentation", "Select one or more augmentations to apply.")
            return
        params = self._collect_params(aug_names)
        augmenters = [build_augmenter(name, params.get(name)) for name in aug_names]
        created = self.data_manager.apply_augmentations(entry_ids, aug_names, augmenters)
        if not created:
            QtWidgets.QMessageBox.warning(self, "Augmentation failed", "No files were augmented.")
            return
        self._refresh_table()
        self.statusBar().showMessage(f"Created {len(created)} augmented file(s)", 3000)

    def _preview_augmentation(self) -> None:
        entry_ids = self._selected_entry_ids()
        if len(entry_ids) != 1:
            QtWidgets.QMessageBox.information(self, "Select one", "Select a single row to preview.")
            return
        aug_names = self._selected_augmentations()
        if not aug_names:
            QtWidgets.QMessageBox.information(self, "No augmentation", "Select one or more augmentations to preview.")
            return
        entry = self.data_manager.find_entry(entry_ids[0])
        if not entry:
            QtWidgets.QMessageBox.warning(self, "Entry missing", "Could not find the selected entry.")
            return
        path = Path(entry.filepath)
        if not path.exists():
            QtWidgets.QMessageBox.warning(self, "File missing", f"File not found: {path}")
            return
        params = self._collect_params(aug_names)
        augmenters = [build_augmenter(name, params.get(name)) for name in aug_names]
        try:
            samples, sample_rate = sf.read(path, always_2d=False)
            samples = samples.astype(np.float32)
            needs_transpose = samples.ndim == 2 and samples.shape[0] > samples.shape[1]
            if needs_transpose:
                samples = samples.T
            augmented = samples
            for augmenter in augmenters:
                augmented = augmenter(samples=augmented, sample_rate=sample_rate)
            if needs_transpose:
                augmented = augmented.T
            workspace_dir = Path.cwd() / "workspace"
            workspace_dir.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                suffix=".wav",
                delete=False,
                dir=str(workspace_dir),
            ) as tmp_file:
                temp_path = Path(tmp_file.name)
            sf.write(temp_path, augmented, sample_rate, format="WAV")
            audio = AudioSegment.from_file(temp_path)
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "Preview failed", str(exc))
            return

        def play_preview() -> None:
            try:
                play(audio)
            finally:
                if temp_path.exists():
                    temp_path.unlink()

        threading.Thread(target=play_preview, daemon=True).start()
        self.statusBar().showMessage("Playing preview...", 3000)

    def _delete_selected(self) -> None:
        entry_ids = self._selected_entry_ids()
        if not entry_ids:
            return
        choice = QtWidgets.QMessageBox.question(
            self,
            "Delete entries",
            "Delete selected entries and remove their files from the workspace?",
            QtWidgets.QMessageBox.StandardButton.Yes,
            QtWidgets.QMessageBox.StandardButton.No,
        )
        if choice == QtWidgets.QMessageBox.StandardButton.Yes:
            self.data_manager.delete_entries(entry_ids, delete_files=True)
            self._refresh_table()
            self.statusBar().showMessage(f"Deleted {len(entry_ids)} entries and files", 3000)

    def _browse_export_dir(self) -> None:
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Select export directory", str(Path.cwd()))
        if directory:
            self.output_path_edit.setText(directory)

    def _export_selected(self) -> None:
        entry_ids = self._selected_entry_ids()
        if not entry_ids:
            entry_ids = [e.id for e in self.data_manager.list_entries()]
        if not entry_ids:
            QtWidgets.QMessageBox.information(self, "Nothing to export", "No entries available.")
            return
        export_format = self.export_format_combo.currentText()
        output_dir = Path(self.output_path_edit.text())
        split_export = self.split_export_checkbox.isChecked()
        train_ratio = self.train_ratio_spin.value() / 100.0
        try:
            export_path = self.data_manager.export(
                entry_ids=entry_ids,
                output_dir=output_dir,
                export_format=export_format,
                copy_audio=self.copy_audio_checkbox.isChecked(),
                split_train_test=split_export,
                train_ratio=train_ratio,
            )
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "Export failed", str(exc))
            return
        QtWidgets.QMessageBox.information(self, "Export complete", f"Exported to {export_path}")
        self.statusBar().showMessage("Export complete", 3000)

    def _save_session(self) -> None:
        try:
            path = self.data_manager.save_to_disk()
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "Save failed", str(exc))
            return
        QtWidgets.QMessageBox.information(self, "Saved", f"Session saved to {path}")
        self.statusBar().showMessage("Session saved", 3000)

    def _handle_item_changed(self, item: QtWidgets.QTableWidgetItem) -> None:
        if self._suppress_table_signals:
            return
        if item.column() == 2:  # transcription
            entry_id = self.table.item(item.row(), 0).data(QtCore.Qt.ItemDataRole.UserRole)
            if isinstance(entry_id, int):
                self.data_manager.update_transcription(entry_id, item.text())

    def _selected_augmentations(self) -> list[str]:
        names: list[str] = []
        for i in range(self.augmentation_list.count()):
            item = self.augmentation_list.item(i)
            if item.checkState() == QtCore.Qt.CheckState.Checked:
                names.append(item.text())
        return names

    def _collect_params(self, names: list[str]) -> dict[str, dict]:
        params: dict[str, dict] = {}
        for name in names:
            widget_map = self.augmentation_param_widgets.get(name, {})
            params[name] = {param: spin.value() for param, spin in widget_map.items()}
        return params

    def _handle_param_focus(self, item: QtWidgets.QListWidgetItem) -> None:
        if not self.params_stack:
            return
        row = self.augmentation_list.row(item)
        if 0 <= row < self.params_stack.count():
            self.params_stack.setCurrentIndex(row)

    def _handle_row_double_clicked(self, row: int, column: int) -> None:  # noqa: ARG002
        item = self.table.item(row, 1)  # filepath column
        if not item:
            return
        path = Path(item.text())
        if not path.exists():
            QtWidgets.QMessageBox.warning(self, "File missing", f"File not found: {path}")
            return
        try:
            audio = AudioSegment.from_file(path)
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "Playback failed", str(exc))
            return

        def play_audio() -> None:
            play(audio)

        threading.Thread(target=play_audio, daemon=True).start()

    def _toggle_split_export_options(self, checked: bool) -> None:
        self.export_format_combo.setEnabled(not checked)
        if checked:
            self.copy_audio_checkbox.setChecked(True)
        self.copy_audio_checkbox.setEnabled(not checked)
        self.train_ratio_spin.setEnabled(checked)

    def _update_export_selection(self) -> None:
        self.export_selection_list.clear()
        entry_ids = self._selected_entry_ids()
        entries = (
            [e for e in self.data_manager.list_entries() if e.id in entry_ids]
            if entry_ids
            else self.data_manager.list_entries()
        )
        for entry in entries:
            label = f"{entry.id}: {entry.filename}"
            if entry.transcription:
                label += f" — {entry.transcription}"
            self.export_selection_list.addItem(label)
