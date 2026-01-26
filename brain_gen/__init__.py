"""Top-level package exports with lazy imports."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_LAZY_IMPORTS = {
    "Omega": "brain_gen.preprocessing.omega",
    "MOUS": "brain_gen.preprocessing.mous",
    "MOUSConditioned": "brain_gen.preprocessing.mous",
    "CamCAN": "brain_gen.preprocessing.camcan",
    "CamCANConditioned": "brain_gen.preprocessing.camcan",
    "TextProcessor": "brain_gen.preprocessing.text",
    "GroupedTextProcessor": "brain_gen.preprocessing.text",
    "ExperimentDL": "brain_gen.training",
    "ExperimentTokenizer": "brain_gen.training",
    "ExperimentTokenizerText": "brain_gen.training",
    "TextBPETokenizerTrainer": "brain_gen.training",
    "ExperimentVidtok": "brain_gen.training",
    "split_datasets": "brain_gen.dataset",
}

__all__ = [
    "Omega",
    "MOUS",
    "MOUSConditioned",
    "CamCAN",
    "CamCANConditioned",
    "TextProcessor",
    "GroupedTextProcessor",
    "TextBPETokenizerTrainer",
    "ExperimentDL",
    "ExperimentVidtok",
    "ExperimentTokenizer",
    "ExperimentTokenizerText",
    "split_datasets",
]


def __getattr__(name: str) -> Any:
    """Lazily import top-level attributes on first access."""
    module_path = _LAZY_IMPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_path)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return the module attribute list for dir()."""
    return sorted(set(list(globals()) + __all__))
