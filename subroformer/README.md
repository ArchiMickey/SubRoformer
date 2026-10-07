[← SubRoformer](../README.md)

# Python package

First complete [source installation](../docs/installation.md).
Run shell examples from the repository root.

| Module or directory | Responsibility |
| --- | --- |
| [model.py](model.py) | `SBRoformer` architecture and frontend construction |
| [model_utils.py](model_utils.py) | OmegaConf loading, YAML inheritance, and model construction |
| [checkpoint_utils.py](checkpoint_utils.py) | Strict checkpoint loading and safetensors export |
| [frontend/](frontend/README.md) | Subband analysis/synthesis and DSP backends |
| [configs/](configs/README.md) | Packaged YAML configs and `get_config_path()` |
| [training/](training/README.md) | Dataset preparation, training, resume, and evaluation |
| [inference/](inference/README.md) | CLI/API separation, streaming outputs, Core ML, and export |

## Imports

```python
from subroformer.model import SBRoformer
from subroformer.configs import get_config_path
from subroformer.model_utils import build_model, load_config
from subroformer.inference.engine import load_separator

config = load_config(get_config_path("sbroformer_base_4stem_silu.yaml"))
```

`build_model(config)` constructs a model from the YAML's `model` section.
`load_separator()` constructs and loads an inference model; see the
[inference guide](inference/README.md#python-api) for audio inputs, outputs,
device selection, and precision.

`subroformer.checkpoint_utils.load_model_checkpoint(model, path)` strictly loads
safetensors or legacy PyTorch weights. For conversion, use
`python -m subroformer.checkpoint_utils`; the
[export guide](inference/README.md#checkpoint-export) describes its arguments.
Released config/checkpoint pairs, hashes, and provenance are recorded in
[checkpoint details](../docs/checkpoints.md).

The desktop app, repository tests, and benchmarks live outside the package.
See the [app guide](../app/README.md), [test guide](../docs/testing.md), and
[historical validation](../docs/validation.md).
