import brain_gen.models.hf_adapters.attn as attn


def test_get_flex_attention_falls_back_on_compile_failure(monkeypatch) -> None:
    monkeypatch.setattr(attn, "_COMPILED_FLEX_ATTENTION", None)
    monkeypatch.setattr(attn.torch.cuda, "is_available", lambda: True)

    def _boom(_fn):
        raise RuntimeError("compile failed")

    monkeypatch.setattr(attn.torch, "compile", _boom)

    fn = attn._get_flex_attention()
    assert fn is attn.flex_attention
