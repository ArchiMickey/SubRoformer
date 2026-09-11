from pathlib import Path

import pytest
import torch

from model import (
    AdditiveFusion,
    BandRMSNorm,
    DEFAULT_GLOBAL_BS_FREQS_PER_BAND,
    GELUFeedForward,
    GEGLUFusion,
    LinearFusion,
    MultiplicativeFusion,
    NoSkipFusion,
    RMSNorm,
    RaggedBandLinear,
    RaggedBandMaskEstimator,
    SiLUFusion,
    SwiGLU,
    TimeMixBlock,
    TransformerBlock,
)
from tests.helpers import tiny_config
from training_utils import build_model, load_config


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("amplitude", [0.0, 1e-4, 1.0, 1000.0])
def test_low_precision_rmsnorm_matches_float32(dtype, amplitude):
    norm = RMSNorm(4).to(dtype)
    inputs = (torch.tensor([[[0.3, -0.3, 1.0, -1.0]]]) * amplitude).to(dtype).requires_grad_()
    reference = RMSNorm(4)(inputs.detach().float())
    with torch.autocast("cpu", dtype=dtype):
        actual = norm(inputs)
    assert actual.dtype == dtype
    assert norm.scale.dtype == dtype
    torch.testing.assert_close(actual.float(), reference, rtol=0.005, atol=1e-4)
    actual.float().sum().backward()
    assert torch.isfinite(inputs.grad).all()
    assert torch.isfinite(norm.scale.grad).all()


def test_transformer_uses_real_swiglu():
    block = TransformerBlock(dim=32, num_heads=4)
    assert isinstance(block.ffn, SwiGLU)
    assert isinstance(block.ffn.act_fn, torch.nn.SiLU)
    assert isinstance(TimeMixBlock(dim=32).act, torch.nn.SiLU)


def test_sbroformer_forward_shape_and_finiteness():
    model = build_model(tiny_config())
    audio = torch.randn(1, 2, 128)
    with torch.inference_mode():
        output = model(audio)
    assert output.shape == audio.shape
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("fusion_type", ["geglu", "silu"])
def test_sbroformer_joint_four_stem_forward_backward(fusion_type):
    config = tiny_config(target_instrument=None)
    config.model.num_stems = 4
    config.model.fusion_type = fusion_type
    model = build_model(config)
    audio = torch.randn(1, 2, 128)

    output = model(audio)
    output.square().mean().backward()

    assert output.shape == (1, 4, 2, 128)
    assert torch.isfinite(output).all()
    assert model.stem_mask_heads is not None
    assert all(head.weight.grad is not None for head in model.stem_mask_heads)


def test_sbroformer_four_stem_config_contract():
    config_path = Path(__file__).parents[1] / "configs" / "sbroformer_base_4stem.yaml"
    config = load_config(str(config_path))

    assert config.model.num_stems == 4
    assert config.training.target_instrument is None
    assert list(config.training.instruments) == ["vocals", "bass", "drums", "other"]
    assert config.training.output_dir == "checkpoints/sbroformer_base_4stem"
    assert config.training.distributed_timeout_seconds == 7200


def _run_forward_backward(config):
    model = build_model(config)
    audio = torch.randn(1, 2, 128, requires_grad=True)
    output = model(audio)
    output.square().mean().backward()
    assert output.shape == audio.shape
    assert torch.isfinite(output).all()
    assert any(parameter.grad is not None for parameter in model.parameters())
    return model


@pytest.mark.parametrize("frontend_type", ["global_stft_bs", "global_stft_mel"])
def test_global_stft_frontends_forward_backward(frontend_type):
    config = tiny_config()
    config.model.frontend_type = frontend_type
    config.model.global_stft_n_fft = 32
    config.model.global_stft_hop_length = 8
    config.model.global_stft_win_length = 32
    config.model.global_band_dim = 32
    config.model.global_zero_dc = False
    if frontend_type == "global_stft_bs":
        config.model.global_freqs_per_bands = [4, 4, 4, 5]
    else:
        config.model.global_mel_bands = 4
    model = _run_forward_backward(config)
    assert model.global_frontend.projection == frontend_type.removeprefix("global_stft_")


def test_one_shot_band_rmsnorm_and_linear_match_band_loop():
    torch.manual_seed(0)
    dim_inputs = (3, 5, 2)
    x = torch.randn(2, 4, 3, 5)
    x[:, :, 0, 3:] = 0
    x[:, :, 2, 2:] = 0
    norm = BandRMSNorm(dim_inputs)
    linear = RaggedBandLinear(dim_inputs, out_features=7)

    actual = linear(norm(x))
    expected_bands = []
    offset = 0
    for band_index, dim_input in enumerate(dim_inputs):
        band = x[:, :, band_index, :dim_input]
        scale = norm.weight[offset:offset + dim_input]
        band = torch.nn.functional.normalize(band, dim=-1, eps=norm.eps) * (dim_input ** 0.5) * scale
        weight = linear.weight[offset:offset + dim_input]
        expected_bands.append(torch.einsum("bti,io->bto", band, weight) + linear.bias[band_index])
        offset += dim_input
    expected = torch.stack(expected_bands, dim=2)
    torch.testing.assert_close(actual, expected)


def test_one_shot_ragged_mask_estimator_matches_band_loop():
    torch.manual_seed(0)
    dim_inputs = (3, 5, 2)
    estimator = RaggedBandMaskEstimator(dim_inputs, dim=7)
    x = torch.randn(2, 4, 3, 7)
    actual = estimator(x)

    expected = x.new_zeros(2, 4, 3, 5)
    offset = 0
    for band_index, dim_input in enumerate(dim_inputs):
        logits = torch.einsum(
            "btd,ird->btir",
            x[:, :, band_index],
            estimator.weight[offset:offset + dim_input],
        ) + estimator.bias[offset:offset + dim_input]
        value, gate = logits.unbind(dim=-1)
        expected[:, :, band_index, :dim_input] = value * gate.sigmoid()
        offset += dim_input
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("frontend_type", ["global_stft_bs", "global_stft_mel"])
def test_global_frontend_mask_aggregation_preserves_complex_identity(frontend_type):
    config = tiny_config()
    config.model.frontend_type = frontend_type
    config.model.global_stft_n_fft = 32
    config.model.global_stft_hop_length = 8
    config.model.global_stft_win_length = 32
    config.model.global_band_dim = 32
    if frontend_type == "global_stft_bs":
        config.model.global_freqs_per_bands = [4, 4, 4, 5]
    else:
        config.model.global_mel_bands = 4
    frontend = build_model(config).global_frontend

    raw_dim = frontend.freq_bins * frontend.audio_channels * 2
    valid = frontend.input_indexes < raw_dim
    band_masks = torch.zeros(1, 3, frontend.n_bands, frontend.max_dim_input)
    band_masks[0, :, valid & (frontend.input_indexes % 2 == 0)] = 1
    complex_mask = frontend.aggregate_masks(band_masks)
    torch.testing.assert_close(complex_mask.real, torch.ones_like(complex_mask.real))
    torch.testing.assert_close(complex_mask.imag, torch.zeros_like(complex_mask.imag))


@pytest.mark.parametrize("patch_factor", [1, 2, 4, 8])
def test_temporal_patch_ablation_forward_backward(patch_factor):
    config = tiny_config()
    config.model.patch_size = [patch_factor, 1]
    model = _run_forward_backward(config)
    assert model.patch_size == (patch_factor, 1)


def test_gradient_checkpointing_forward_backward():
    config = tiny_config()
    config.model.gradient_checkpointing = True
    model = _run_forward_backward(config)
    assert model.gradient_checkpointing is True


@pytest.mark.parametrize("analysis_type", ["stft", "cnn", "direct"])
def test_local_analysis_ablation_forward_backward(analysis_type):
    config = tiny_config()
    config.model.analysis_type = analysis_type
    config.model.dim_sp = 32
    model = _run_forward_backward(config)
    assert model.analysis_type == analysis_type
    assert model.feature_dim == 32


def test_local_analysis_ablation_uses_same_feature_grid():
    audio = torch.randn(1, 2, 128)
    shapes = []
    for analysis_type in ["stft", "cnn", "direct"]:
        config = tiny_config()
        config.model.analysis_type = analysis_type
        config.model.hop_length = 2
        config.model.direct_frame_size = 2
        config.model.analysis_centered = True
        model = build_model(config)
        padded = model.pad_audio(audio, model.sb_factor)
        subbands = model.sb_filter.analysis(padded)
        shapes.append(model.analysis(subbands).shape)
    assert len(set(shapes)) == 1


@pytest.mark.parametrize(
    ("fusion_type", "fusion_class"),
    [
        ("geglu", GEGLUFusion),
        ("silu", SiLUFusion),
        ("add", AdditiveFusion),
        ("multiply", MultiplicativeFusion),
        ("none", NoSkipFusion),
        ("linear", LinearFusion),
    ],
)
def test_skip_fusion_ablation_forward_backward(fusion_type, fusion_class):
    config = tiny_config()
    config.model.fusion_type = fusion_type
    config.model.use_fusion = None
    model = _run_forward_backward(config)
    assert isinstance(model.fusion, fusion_class)


def test_silu_fusion_preserves_checkpoint_layout_and_uses_silu():
    fusion = SiLUFusion(dim=4)
    fusion.load_state_dict(GEGLUFusion(dim=4).state_dict(), strict=True)
    assert isinstance(fusion.act, torch.nn.SiLU)
    decoder = torch.randn(1, 4, 3, 5)
    encoder = torch.randn_like(decoder)
    combined = torch.cat([fusion.norm_dec(decoder), fusion.norm_enc(encoder)], dim=1)
    gate, value = fusion.proj_in(combined).chunk(2, dim=1)
    expected = decoder + fusion.proj_out(torch.nn.functional.silu(gate) * value)
    torch.testing.assert_close(fusion(decoder, encoder), expected)


def test_multiplicative_fusion_uses_direct_elementwise_product():
    fusion = MultiplicativeFusion(dim=4)
    x_dec = torch.randn(2, 4, 3, 5)
    x_enc = torch.randn(2, 4, 3, 5)

    torch.testing.assert_close(fusion(x_dec, x_enc), x_dec * x_enc)
    assert sum(parameter.numel() for parameter in fusion.parameters()) == 0


def test_additive_fusion_uses_direct_elementwise_sum():
    fusion = AdditiveFusion(dim=4)
    x_dec = torch.randn(2, 4, 3, 5)
    x_enc = torch.randn(2, 4, 3, 5)

    torch.testing.assert_close(fusion(x_dec, x_enc), x_dec + x_enc)
    assert sum(parameter.numel() for parameter in fusion.parameters()) == 0


def test_no_skip_fusion_returns_decoder_features():
    fusion = NoSkipFusion(dim=4)
    x_dec = torch.randn(2, 4, 3, 5)
    x_enc = torch.randn(2, 4, 3, 5)

    torch.testing.assert_close(fusion(x_dec, x_enc), x_dec)
    assert sum(parameter.numel() for parameter in fusion.parameters()) == 0


def test_linear_fusion_projects_concatenated_features():
    fusion = LinearFusion(dim=4)
    x_dec = torch.randn(2, 4, 3, 5)
    x_enc = torch.randn(2, 4, 3, 5)

    expected = fusion.linear(torch.cat([x_dec, x_enc], dim=1))
    torch.testing.assert_close(fusion(x_dec, x_enc), expected)


def test_gelu_ffn_ablation_forward_backward():
    config = tiny_config()
    config.model.ffn_type = "gelu"
    model = _run_forward_backward(config)
    assert isinstance(model.blocks[0].time_block.ffn, GELUFeedForward)
    assert isinstance(model.blocks[0].freq_block.ffn, GELUFeedForward)
    assert isinstance(model.pre_blocks[0].time_block.act, torch.nn.GELU)
    assert isinstance(model.post_blocks[0].time_block.act, torch.nn.GELU)
