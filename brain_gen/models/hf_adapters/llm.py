import torch
from torch import nn

try:  # optional HF deps
    from transformers.models.smollm3.configuration_smollm3 import (  # noqa: F401
        SmolLM3Config,
    )
    from transformers.models.smollm3.modeling_smollm3 import SmolLM3Model  # noqa: F401
except Exception:  # pragma: no cover
    SmolLM3Config = None
    SmolLM3Model = None

try:
    from transformers.models.minimax.configuration_minimax import (  # noqa: F401
        MiniMaxConfig,
    )
    from transformers.models.minimax.modeling_minimax import MiniMaxModel  # noqa: F401
except Exception:  # pragma: no cover
    MiniMaxConfig = None
    MiniMaxModel = None
from .masking import _block_causal_mask


class GenericModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        kwargs.pop("reduced_shape")
        self.block_size = kwargs.pop("block_size", 1)

        config_class = globals()[kwargs.pop("config_class")]
        self.config = config_class(**kwargs)

        model_class = globals()[kwargs.pop("model_class")]
        self.model = model_class(self.config)

    def forward(
        self,
        x: torch.Tensor | None = None,
        *,
        inputs_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values=None,
        use_cache: bool | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if (x is None) == (inputs_embeds is None):
            raise ValueError("Provide exactly one of `x` or `inputs_embeds`.")

        if attention_mask is not None:
            seq_len = x.shape[1] if x is not None else inputs_embeds.shape[1]
            attention_mask = attention_mask[:, :seq_len]

        if self.block_size > 1:
            main_input = x if x is not None else inputs_embeds
            attention_mask = _block_causal_mask(
                self.config,
                self.block_size,
                past_key_values,
                main_input,
                include_position_ids=False,
            )

        model_kwargs = dict(
            input_ids=x,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_hidden_states=False,
            return_dict=True,
        )
        model_kwargs.update(kwargs)

        outputs = self.model(**model_kwargs)

        cache = outputs.past_key_values if use_cache else None
        if use_cache or past_key_values is not None:
            return outputs.last_hidden_state, cache
        return outputs.last_hidden_state

    def get_embed_layer(self) -> nn.Module:
        return self.model.embed_tokens


class SmoLLM3(GenericModel):
    def __init__(self, *args, **kwargs):
        kwargs["config_class"] = "SmolLM3Config"
        kwargs["model_class"] = "SmolLM3Model"
        super().__init__(*args, **kwargs)


class MiniMax(GenericModel):
    def __init__(self, *args, **kwargs):
        kwargs["config_class"] = "MiniMaxConfig"
        kwargs["model_class"] = "MiniMaxModel"
        super().__init__(*args, **kwargs)


class xLSTM(GenericModel):
    def __init__(self, *args, **kwargs):
        kwargs["config_class"] = "xLSTMConfig"
        kwargs["model_class"] = "xLSTMModel"
        super().__init__(*args, **kwargs)

    def get_embed_layer(self) -> nn.Module:
        return self.model.embeddings


class FalconH1(GenericModel):
    def __init__(self, *args, **kwargs):
        kwargs["config_class"] = "FalconH1Config"
        kwargs["model_class"] = "FalconH1Model"
        super().__init__(*args, **kwargs)


class FalconMamba(GenericModel):
    def __init__(self, *args, **kwargs):
        kwargs["config_class"] = "FalconMambaConfig"
        kwargs["model_class"] = "FalconMambaModel"
        super().__init__(*args, **kwargs)

    def get_embed_layer(self) -> nn.Module:
        return self.model.embeddings


class Gemma3(GenericModel):
    def __init__(self, *args, **kwargs):
        kwargs["config_class"] = "Gemma3TextConfig"
        kwargs["model_class"] = "Gemma3TextModel"
        super().__init__(*args, **kwargs)


class BLT(GenericModel):
    def __init__(self, *args, **kwargs):
        kwargs["config_class"] = "BltConfig"
        kwargs["model_class"] = "BltModel"
        super().__init__(*args, **kwargs)

    def get_embed_layer(self) -> nn.Module:
        return self.model.get_input_embeddings()
