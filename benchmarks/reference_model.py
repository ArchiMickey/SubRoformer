from dsp.subband_fast_triangle import SubbandFilter
from model_utils import build_model


def build_reference_model(config):
    """Restore the unchanged legacy DSP implementation for pre-merge comparisons."""
    model = build_model(config)
    source = model.sb_filter
    if source is not None:
        reference = SubbandFilter(
            source.sr,
            source.banks,
            source.factor,
            chunk_size=source.chunk_size,
            bandpass_filter_len=source.bandpass_filter_len,
            upsample_filter_len=source.upsample_filter_len,
        )
        reference.load_state_dict(source.state_dict())
        model.sb_filter = reference
    return model
