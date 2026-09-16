import pytest
import torch

from dsp.inference_optimized import InferenceSubbandFilter
from dsp2 import SubbandFilter
from inference_optimized import optimize_for_inference
from tests.helpers import tiny_config
from training_utils import build_model


@pytest.mark.parametrize("num_stems", [1, 4])
def test_dsp2_checkpoint_forward_backward_and_step(num_stems):
    config = tiny_config()
    config.model.num_stems = num_stems
    original = build_model(config)
    assert isinstance(original.sb_filter, InferenceSubbandFilter)
    config.model.dsp_backend = "dsp2"
    model = build_model(config)
    assert isinstance(model.sb_filter, SubbandFilter)
    model.load_state_dict(original.state_dict(), strict=True)
    original.load_state_dict(model.state_dict(), strict=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    audio = torch.randn(1, 2, 129, requires_grad=True)
    output = model(audio)
    expected_shape = (1, 2, 129) if num_stems == 1 else (1, num_stems, 2, 129)
    assert output.shape == expected_shape
    loss = output.square().mean()
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(audio.grad).all()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
    optimizer.step()
    model.eval()
    with torch.inference_mode():
        assert torch.isfinite(model(audio.detach())).all()
    with pytest.raises(ValueError, match="use dsp2 directly"):
        optimize_for_inference(model)


@pytest.mark.parametrize("backend,frontend", [
    ("invalid", "waveform_subband"),
    ("dsp2", "global_stft_bs"),
    ("dsp2", "global_stft_mel"),
])
def test_invalid_backend_combinations(backend, frontend):
    config = tiny_config()
    config.model.dsp_backend = backend
    config.model.frontend_type = frontend
    with pytest.raises(ValueError, match="dsp_backend"):
        build_model(config)
