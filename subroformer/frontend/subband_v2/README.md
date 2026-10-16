[← Frontend guide](../README.md)

# Subband V2

`SubbandFilterV2` provides complex FIR analysis and synthesis with an explicit
band count, Kaiser or Hamming filter windows, retained boundary padding, and
float32/complex64 arithmetic by default. It uses DSP3's fixed-point frequency
modulation helper. The API is separate from the legacy real-filterbank
implementations in `subroformer/frontend/subband/dsp*`.

```python
import torch
from subroformer.frontend.subband_v2 import SubbandFilterV2

filterbank = SubbandFilterV2(
    sr=48000,
    bins=64,
    factor=10,
    max_half_bandwidth=2300,
).eval()
audio = torch.randn(1, 2, 96000)
subbands = filterbank.analysis(audio)
reconstruction = filterbank.synthesis(subbands, length=audio.shape[-1])
```

Analysis returns complex `[batch, channels, bins, frames]` subbands. Synthesis
accepts those padded subbands and can restore the requested original length.
Use `mel_linear_centers(sr, bins, max_half_bandwidth)` to inspect the exact
band centers.

This filterbank is not selected through `model.dsp_backend` and does not
replace the V1 analysis/synthesis contract used by existing SBRoformer
checkpoints.
