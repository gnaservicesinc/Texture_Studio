"""Native encoder backend selection and narrowly scoped dependency notices."""
from __future__ import annotations

import os
from pathlib import Path
import sys
from types import SimpleNamespace
import warnings

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import frozen_dino_height as dino


@pytest.mark.parametrize("device", ("cpu", "mps"))
@pytest.mark.parametrize("previous", (None, "custom-value"))
def test_native_import_selects_pytorch_and_keeps_real_warnings(monkeypatch, tmp_path, device, previous):
    if previous is None:
        monkeypatch.delenv("XFORMERS_DISABLED", raising=False)
    else:
        monkeypatch.setenv("XFORMERS_DISABLED", previous)
    layers = {}
    for name in ("attention", "block", "swiglu_ffn"):
        layers[name] = SimpleNamespace(XFORMERS_AVAILABLE=True)
        monkeypatch.setitem(sys.modules, "dinov2.layers." + name, layers[name])
    unrelated = SimpleNamespace(XFORMERS_AVAILABLE=True)
    monkeypatch.setitem(sys.modules, "unrelated_attention", unrelated)
    official = object()

    def import_encoder(name):
        assert name == "dinov2.models.vision_transformer"
        assert sys.path[0] == str(tmp_path)
        assert os.environ["XFORMERS_DISABLED"] == "1"
        for module, label in (("swiglu_ffn", "SwiGLU"), ("attention", "Attention"), ("block", "Block")):
            for state in ("not available", "disabled"):
                warnings.warn_explicit(f"xFormers is {state} ({label})", UserWarning,
                                      "vendor.py", 1, module="dinov2.layers." + module)
        warnings.warn_explicit("unexpected encoder warning", UserWarning,
                              "vendor.py", 2, module="dinov2.layers.attention")
        warnings.warn_explicit("xFormers is not available (Attention)", RuntimeWarning,
                              "vendor.py", 3, module="dinov2.layers.attention")
        warnings.warn_explicit("xFormers is not available (Attention)", UserWarning,
                              "other.py", 4, module="other_library.attention")
        return official

    monkeypatch.setattr(dino.importlib, "import_module", import_encoder)
    original_path = list(sys.path)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert dino.import_official_encoder(tmp_path, torch.device(device)) is official
        # The import filter must not escape into the rest of the training run.
        warnings.warn_explicit("xFormers is not available (Attention)", UserWarning,
                              "vendor.py", 5, module="dinov2.layers.attention")
    assert len(caught) == 4
    assert str(caught[0].message) == "unexpected encoder warning"
    assert caught[1].category is RuntimeWarning
    assert caught[2].filename == "other.py"
    assert caught[3].lineno == 5
    assert all(not layer.XFORMERS_AVAILABLE for layer in layers.values())
    assert unrelated.XFORMERS_AVAILABLE
    assert sys.path == original_path
    assert os.environ.get("XFORMERS_DISABLED") == previous


@pytest.mark.parametrize("previous", (None, "already-disabled"))
def test_failed_import_restores_environment_and_search_path(monkeypatch, tmp_path, previous):
    if previous is None:
        monkeypatch.delenv("XFORMERS_DISABLED", raising=False)
    else:
        monkeypatch.setenv("XFORMERS_DISABLED", previous)

    def broken_import(_name):
        assert os.environ["XFORMERS_DISABLED"] == "1"
        raise ImportError("real dependency failure")

    monkeypatch.setattr(dino.importlib, "import_module", broken_import)
    original_path = list(sys.path)
    with pytest.raises(ImportError, match="real dependency failure"):
        dino.import_official_encoder(tmp_path, torch.device("mps"))
    assert sys.path == original_path
    assert os.environ.get("XFORMERS_DISABLED") == previous


def test_other_accelerators_keep_upstream_backend_and_notices(monkeypatch, tmp_path):
    monkeypatch.setenv("XFORMERS_DISABLED", "user-setting")
    attention = SimpleNamespace(XFORMERS_AVAILABLE=True)
    monkeypatch.setitem(sys.modules, "dinov2.layers.attention", attention)
    official = object()

    def import_encoder(_name):
        assert os.environ["XFORMERS_DISABLED"] == "user-setting"
        warnings.warn_explicit("xFormers is not available (Attention)", UserWarning,
                              "vendor.py", 1, module="dinov2.layers.attention")
        return official

    monkeypatch.setattr(dino.importlib, "import_module", import_encoder)
    with pytest.warns(UserWarning, match="xFormers is not available"):
        assert dino.import_official_encoder(tmp_path, torch.device("cuda")) is official
    assert attention.XFORMERS_AVAILABLE
    assert os.environ["XFORMERS_DISABLED"] == "user-setting"
