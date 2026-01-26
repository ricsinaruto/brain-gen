from importlib import import_module

_MODEL_CLASS_PATHS = {
    "NTD": "brain_gen.models.ntd.NTD",
    "BENDRForecast": "brain_gen.models.bendr.BENDRForecast",
    "WavenetFullChannel": "brain_gen.models.wavenet.WavenetFullChannel",
    "Wavenet3D": "brain_gen.models.wavenet.Wavenet3D",
    "TASA3D": "brain_gen.models.tasa3d.TASA3D",
    "ClassifierContinuous": "brain_gen.models.classifier.ClassifierContinuous",
    "ClassifierQuantized": "brain_gen.models.classifier.ClassifierQuantized",
    "ClassifierQuantizedImage": "brain_gen.models.classifier.ClassifierQuantizedImage",
    "BrainOmniCausalTokenizer": "brain_gen.models.brainomni.BrainOmniCausalTokenizer",
    "BrainOmniCausalTokenizerSEANetChannelMix": "brain_gen.models.tokenizers.braintokmix.BrainOmniCausalTokenizerSEANetChannelMix",
    "BrainOmniCausalForecast": "brain_gen.models.brainomni.BrainOmniCausalForecast",
    "FlatGPT": "brain_gen.models.flatgpt.FlatGPT",
    "FlatGPTEmbeds": "brain_gen.models.flatgpt.FlatGPTEmbeds",
    "FlatGPTRVQ": "brain_gen.models.flatgpt.FlatGPTRVQ",
    "FlatGPTMix": "brain_gen.models.flatgpt.FlatGPTMix",
    "FlatGPTMixRVQ": "brain_gen.models.flatgpt.FlatGPTMixRVQ",
    "FlatGPTEmbedsRVQ": "brain_gen.models.flatgpt.FlatGPTEmbedsRVQ",
    "STGPT2MEG": "brain_gen.models.stgpt2meg.STGPT2MEG",
    "Vidtok": "brain_gen.models.tokenizers.vidtok.Vidtok",
    "VidtokRVQ": "brain_gen.models.tokenizers.vidtok.VidtokRVQ",
}

_LOSS_CLASS_PATHS = {
    "CrossEntropy": "brain_gen.losses.classification.CrossEntropy",
    "CrossEntropyWeighted": "brain_gen.losses.classification.CrossEntropyWeighted",
    "MSE": "brain_gen.losses.reconstruction.MSE",
    "CrossEntropyWithCodes": "brain_gen.losses.classification.CrossEntropyWithCodes",
    "NTDLoss": "brain_gen.losses.diffusion.NTDLoss",
    "VQNSPLoss": "brain_gen.losses.reconstruction.VQNSPLoss",
    "NLL": "brain_gen.losses.reconstruction.NLL",
    "VQVAELoss": "brain_gen.losses.reconstruction.VQVAELoss",
    "ChronoFlowLoss": "brain_gen.losses.reconstruction.ChronoFlowLoss",
    "CrossEntropyBalanced": "brain_gen.losses.classification.CrossEntropyBalanced",
    "CrossEntropyMasked": "brain_gen.losses.classification.CrossEntropyMasked",
    "CrossEntropySpectral": "brain_gen.losses.classification.CrossEntropySpectral",
    "SpectralLoss": "brain_gen.losses.reconstruction.SpectralLoss",
    "BrainOmniCausalTokenizerLoss": "brain_gen.losses.brainomni.BrainOmniCausalTokenizerLoss",
    "BrainOmniCausalForecastLoss": "brain_gen.losses.brainomni.BrainOmniCausalForecastLoss",
    "VQVAEHF": "brain_gen.losses.reconstruction.VQVAEHF",
    "VidtokLoss": "brain_gen.losses.vidtok.VidtokLoss",
}


def _load_from_path(dotted_path: str):
    """Resolve a dotted path to an object, importing lazily."""
    module_path, attr = dotted_path.rsplit(".", 1)
    module = import_module(module_path)
    return getattr(module, attr)


def get_model_class(model_name: str):
    """Get model class by name."""
    try:
        return _load_from_path(_MODEL_CLASS_PATHS[model_name])
    except KeyError as exc:
        raise ValueError(f"Unknown model name: {model_name}") from exc


def get_loss_class(loss_name: str):
    """Get loss class by name."""
    try:
        return _load_from_path(_LOSS_CLASS_PATHS[loss_name])
    except KeyError as exc:
        raise ValueError(f"Unknown loss name: {loss_name}") from exc
