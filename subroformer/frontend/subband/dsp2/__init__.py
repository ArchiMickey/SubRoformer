"""Opt-in fixed-point subband filtering; the original dsp package is unchanged."""

from ..dsp.banks import mel_linear_banks_triangle

from .subband_fast_triangle import SubbandFilter

__all__ = ["SubbandFilter", "mel_linear_banks_triangle"]
