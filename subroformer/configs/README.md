[← Package guide](../README.md)

# Model configs

These YAMLs configure the model, audio, training, augmentation, validation, and
inference settings through OmegaConf. Run shell examples from the repository
root.

## Inventory

| Config | Intended model or use |
| --- | --- |
| [sbroformer_base_4stem.yaml](sbroformer_base_4stem.yaml) | Original Base four-stem model, GEGLU fusion |
| [sbroformer_base_4stem_silu.yaml](sbroformer_base_4stem_silu.yaml) | Default perceptual-finetuned four-stem model, SiLU fusion, DSP3 |
| [sbroformer_base_4stem_silu_dsp2.yaml](sbroformer_base_4stem_silu_dsp2.yaml) | Matching SiLU model with explicit DSP2 for comparison |
| [sbroformer_base_vocals.yaml](sbroformer_base_vocals.yaml) | Single-stem vocals training |
| [sbroformer_base_bass.yaml](sbroformer_base_bass.yaml) | Single-stem bass training |
| [sbroformer_base_drums.yaml](sbroformer_base_drums.yaml) | Single-stem drums training |
| [sbroformer_base_other.yaml](sbroformer_base_other.yaml) | Single-stem other training |

Keep each checkpoint paired with its matching model config; see
[released checkpoint pairs](../../docs/checkpoints.md). The supplied joint
checkpoint cannot be used with single-stem configs; no single-stem weights are
included. See the [training guide](../training/README.md#training) for dataset
and batching requirements, and the [frontend guide](../frontend/README.md) for
backend selection.

## Installed resources and inheritance

Use `get_config_path(name)` to locate a packaged YAML independently of the
working directory:

```python
from subroformer.configs import get_config_path
from subroformer.model_utils import load_config

config = load_config(get_config_path("sbroformer_base_4stem_silu.yaml"))
```

The seven supplied configs are self-contained. Custom YAMLs may use a top-level
`extends` field. `load_config()` recursively loads the base and merges the
child's settings over it; relative base paths are resolved beside the child
YAML, and absolute paths are accepted.

## Inference resolution

Without a config path, PyTorch inference uses the packaged
`sbroformer_base_4stem_silu.yaml`. An existing explicit config file takes
precedence; a missing explicit path retains the canonical Hub fallback.
Core ML uses the config bundled with its artifact, and custom config/checkpoint
pairs require PyTorch.

Full checkpoint/cache/offline resolution rules are in the
[inference guide](../inference/README.md#cli-and-model-resolution).
Local YAMLs live under `subroformer/configs/`; Hub identifiers retain their
`configs/...` paths.
