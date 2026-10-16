# Subband V2 provenance

`subband_v2.py` is copied from the local AudioSSL extension at
`/home/archimickey/Projects/AudioSSL/subband_pytorch/subband_v2.py`.
AudioSSL checkout HEAD was `43237483135ef7112a9e58181a967374886d8677`;
the source file was untracked there and had SHA256
`b8d154696dc519f7d510ba2935eaba47e4642a15cc30ea821722b241e4f938ac`.
Its filterbank algorithm and public API are preserved. The frequency-shift
helper imports this repository's equivalent implementation from
`subroformer.frontend.subband.dsp3.utils`.

The adjacent MIT license is retained from the AudioSSL vendored
`subband_pytorch` package.
