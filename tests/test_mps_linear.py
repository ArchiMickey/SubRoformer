import pytest
import torch

from subroformer.model import MPSCompatibleLinear


@pytest.mark.parametrize("device", ["cpu", "mps"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("shape", [(2, 7), (2, 3, 7)])
@pytest.mark.parametrize("bias", [False, True])
def test_linear_preserves_checkpoint_dtype_outputs_and_gradients(device, dtype, shape, bias):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")
    torch.manual_seed(0)
    reference = torch.nn.Linear(7, 5, bias=bias)
    layer = MPSCompatibleLinear(7, 5, bias=bias).to(device=device, dtype=dtype)
    layer.load_state_dict(reference.state_dict(), strict=True)
    reference.load_state_dict({name: value.float().cpu() for name, value in layer.state_dict().items()})
    inputs = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
    expected_inputs = inputs.detach().float().cpu().requires_grad_()
    expected = reference(expected_inputs)
    actual = layer(inputs)
    assert actual.dtype == dtype
    assert layer.weight.dtype == dtype
    assert list(layer.state_dict()) == list(reference.state_dict())
    torch.testing.assert_close(actual.float().cpu(), expected, rtol=0.01, atol=0.002)
    actual.float().sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(inputs.grad.float().cpu(), expected_inputs.grad, rtol=0.01, atol=0.002)
    torch.testing.assert_close(layer.weight.grad.float().cpu(), reference.weight.grad, rtol=0.01, atol=0.002)
    if bias:
        torch.testing.assert_close(layer.bias.grad.float().cpu(), reference.bias.grad)


@pytest.mark.parametrize("weight_dtype", [torch.float16, torch.float32])
def test_mps_bias_is_not_delegated_to_fused_linear(monkeypatch, weight_dtype):
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")
    original = torch.nn.functional.linear

    def without_fused_bias(inputs, weight, bias=None):
        assert bias is None
        return original(inputs, weight)

    monkeypatch.setattr(torch.nn.functional, "linear", without_fused_bias)
    layer = MPSCompatibleLinear(341, 128).to(device="mps", dtype=weight_dtype)
    with torch.no_grad():
        layer.bias.copy_(torch.linspace(0.1, 0.9, 128, device="mps", dtype=weight_dtype))
    inputs = torch.zeros(2, 3, 341, device="mps")
    with torch.autocast("mps", dtype=torch.float16):
        output = layer(inputs)
    assert output.dtype == torch.float16
    torch.testing.assert_close(output, layer.bias.half().expand_as(output), rtol=0, atol=0)
