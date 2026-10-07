[← SubRoformer](../README.md)

# Tests

First complete [source installation](installation.md). Run commands from the
repository root. The broader evaluation/test imports require `ffmpeg` and
`ffprobe` on PATH (`brew install ffmpeg` on macOS).

```bash
uv run pytest
uv run torchrun --standalone --nproc-per-node=2 --module tests.ddp_data_smoke
RANDOM_MIXING=0 uv run torchrun --standalone --nproc-per-node=2 --module tests.ddp_data_smoke
uv run torchrun --standalone --nproc-per-node=2 --module tests.ddp_evaluate_smoke
```

See [preparation validation](validation.md) for the checks performed.
Tests use synthetic audio. This public recipe is not a reproduction of the
supplied checkpoint's extra-data training run.

For GUI tests, see [app tests](../app/README.md#tests).
