[← SubRoformer](../README.md)

# Source installation

Run these commands from the repository root. For desktop installation and launch,
see the [app guide](../app/README.md#install-and-launch).

Use Python 3.11 and a PyTorch installation appropriate for your CPU, CUDA, or Metal
environment. A source checkout is recommended for access to tests and benchmarks;
YAML configs and native kernel sources are included in the installed package.

```bash
uv venv --python 3.11
uv pip install --python .venv/bin/python -e '.[test]'
```

Torch attention is the default; FlashAttention is optional. No sibling
repository, private data adapter, or local dataset mount is required.

The implementation lives in `subroformer/`, with DSP backends under `frontend/`,
training and evaluation under `training/`, and inference under `inference/`.
Usage guides live beside their package modules as `README.md` files.
The desktop app, tests, and benchmarks remain at the repository root;
installation, checkpoint provenance, testing, and historical reports live in
`docs/`. See the [package guide](../subroformer/README.md) for imports and module
responsibilities, and the [config guide](../subroformer/configs/README.md) for
installed YAML resources and inheritance.

Use `python -m subroformer.training.train`,
`python -m subroformer.evaluate`, and `python -m subroformer.inference` for the
CLI tools. Old top-level imports and script paths have been removed.
`python -m app` remains the source desktop launcher.

Historical benchmark JSONs and DSP3 source manifests retain their original
paths and hashes. Those paths describe the recorded source revision; current
DSP files live under `subroformer/frontend/`.

Next: [training](../subroformer/training/README.md), [evaluation](../subroformer/training/README.md#evaluation),
[inference](../subroformer/inference/README.md), or [tests](testing.md).
