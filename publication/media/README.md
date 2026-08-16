# ECG Trust Lab demo media

This directory contains publication-oriented media assembled from the project's
already verified browser captures. It does not modify or replace any scientific
result, model, release artifact, or audit record.

`ecg-trust-lab-research-demo.mp4` is a silent, approximately 30-second overview
of the local research interface. It shows the ready state, calibrated five-class
probabilities, the frozen accept/defer gate, all 12 ECG leads, and a Grad-CAM
overlay. Every rendered segment states or preserves the research-only scope.

The exact source-capture hashes, playback properties, output hashes, and decode
check are recorded in `demo-video-manifest.json`. `build_demo_video.py` is the
local rebuild script; its `--ffmpeg` argument accepts any compatible FFmpeg
executable and regenerates the transient `slides/` build directory.

Rebuilding requires Python 3.12, Pillow, FFmpeg with H.264 encoding support,
and the two verified browser captures named in the manifest. One reproducible
invocation shape is:

```powershell
uv run --with pillow python publication/media/build_demo_video.py `
  --initial <initial-browser-capture.png> `
  --gradcam <gradcam-browser-capture.png> `
  --ffmpeg <ffmpeg-executable> `
  --output-dir publication/media
```

The waveform imagery is derived from PTB-XL 1.0.3, provided by Wagner et al.
through PhysioNet under the Creative Commons Attribution 4.0 International
license. Raw PTB-XL files are not included. See the repository
[data card](../../docs/DATA_CARD.md) for the full dataset citation, DOI,
license, and integrity record.

This prototype is not a medical device and must not be used for diagnosis,
treatment, triage, emergency decisions, or other clinical care.
