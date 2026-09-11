[← SubRoformer](../README.md)

# Source installation

Run these commands from the repository root. For desktop installation and launch,
see the [app guide](../app/README.md#install-and-launch).

Use Python 3.11 and a PyTorch installation appropriate for your CPU, CUDA, or Metal
environment. A source checkout is recommended for access to configs and tests.

```bash
uv venv --python 3.11
uv pip install --python .venv/bin/python -e '.[test]'
```

Torch attention is the default; FlashAttention is optional. No sibling
repository, private data adapter, or local dataset mount is required.

Next: [training](training.md), [evaluation](evaluation.md),
[inference](inference.md), or [tests](testing.md).
