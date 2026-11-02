from types import SimpleNamespace

import torch
import torch.nn.functional as F

from subroformer.training.training_utils import fast_sdr, multiresolution_stft_l1, separation_loss


def test_mrstft_matches_zfturbo_complex_l1_sum():
    torch.manual_seed(0)
    output = torch.randn(1, 2, 256)
    target = torch.randn(1, 2, 256)
    window_sizes = [64, 32]

    actual = multiresolution_stft_l1(
        output,
        target,
        window_sizes=window_sizes,
        n_fft=64,
        hop_size=10,
        normalized=False,
    )

    expected = output.new_zeros(())
    for window_size in window_sizes:
        window = torch.hann_window(window_size)
        kwargs = {
            "n_fft": max(window_size, 64),
            "hop_length": 10,
            "win_length": window_size,
            "window": window,
            "normalized": False,
            "return_complex": True,
        }
        output_stft = torch.stft(output.flatten(0, 1), **kwargs)
        target_stft = torch.stft(target.flatten(0, 1), **kwargs)
        expected = expected + F.l1_loss(output_stft, target_stft)

    torch.testing.assert_close(actual, expected)


def test_four_stem_separation_loss_matches_equal_stem_average_and_backward():
    torch.manual_seed(1)
    output = torch.randn(2, 4, 2, 256, requires_grad=True)
    target = torch.randn(2, 4, 2, 256)
    weight = 0.25
    config = SimpleNamespace(
        training={
            "loss": "l1_wav_l1_multistft",
            "multistft_window_sizes": [64, 32],
            "multistft_n_fft": 64,
            "multistft_hop_size": 10,
            "multistft_normalized": False,
            "multistft_weight": weight,
        }
    )

    actual = separation_loss(output, target, config)
    per_stem = []
    for stem in range(output.shape[1]):
        waveform = F.l1_loss(output[:, stem], target[:, stem])
        mrstft = multiresolution_stft_l1(
            output[:, stem],
            target[:, stem],
            window_sizes=[64, 32],
            n_fft=64,
            hop_size=10,
            normalized=False,
        )
        per_stem.append(waveform + weight * mrstft)
    expected = torch.stack(per_stem).mean()

    torch.testing.assert_close(actual, expected)
    assert torch.isfinite(actual)
    actual.backward()
    assert output.grad is not None
    assert torch.isfinite(output.grad).all()


def test_fast_sdr_averages_slices_and_handles_perfect_estimates():
    target = torch.ones(2, 2, 8)
    output = target.clone()
    output[0] *= 0.5
    score = fast_sdr(output, target)
    assert torch.isfinite(score)
    expected = (20 * torch.log10(torch.tensor(2.0)) +
                -10 * torch.log10(torch.tensor(torch.finfo(torch.float32).tiny / 8))) / 2
    torch.testing.assert_close(score, expected)
    assert fast_sdr(torch.zeros_like(target), torch.zeros_like(target)) == 0
