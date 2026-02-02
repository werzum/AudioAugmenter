from __future__ import annotations

from typing import Dict

from audiomentations import (
    AddGaussianNoise,
    PitchShift,
    TimeStretch,
)

DEFAULT_PARAMS: dict[str, dict] = {
    "gaussian_noise": {"min_amplitude": 0.01, "max_amplitude": 0.1},
    "time_stretch": {"min_rate": 0.8, "max_rate": 1.25},
    "pitch_shift": {"min_semitones": -3.0, "max_semitones": 3.0},
}


def build_augmenter(name: str, params: dict | None = None):
    """Build a single augmentation instance with optional parameter overrides."""
    params = params or {}
    if name == "gaussian_noise":
        return AddGaussianNoise(
            min_amplitude=float(params.get("min_amplitude", DEFAULT_PARAMS[name]["min_amplitude"])),
            max_amplitude=float(params.get("max_amplitude", DEFAULT_PARAMS[name]["max_amplitude"])),
            p=1.0,
        )
    if name == "time_stretch":
        return TimeStretch(
            min_rate=float(params.get("min_rate", DEFAULT_PARAMS[name]["min_rate"])),
            max_rate=float(params.get("max_rate", DEFAULT_PARAMS[name]["max_rate"])),
            p=1.0,
        )
    if name == "pitch_shift":
        return PitchShift(
            min_semitones=float(params.get("min_semitones", DEFAULT_PARAMS[name]["min_semitones"])),
            max_semitones=float(params.get("max_semitones", DEFAULT_PARAMS[name]["max_semitones"])),
            p=1.0,
        )
    raise ValueError(f"Unknown augmentation: {name}")


def build_augmentations(params_by_name: Dict[str, dict] | None = None) -> Dict[str, object]:
    """Build augmentation instances for all known names."""
    params_by_name = params_by_name or {}
    return {name: build_augmenter(name, params_by_name.get(name)) for name in DEFAULT_PARAMS.keys()}


def list_augmentation_names() -> list[str]:
    return list(DEFAULT_PARAMS.keys())
