# CLAUDE.md — ChatterBox-Lightning

Drop-in optimized fork of Chatterbox TTS (resemble-ai/chatterbox). 30-50% faster inference via KV cache pre-allocation, BF16 Flash SDPA, and torch.compile on the GPT-2 backbone. No model weights changed.

## Structure

```
ChatterBox-Lightning/
├── src/                        # Chatterbox source (from upstream)
├── example_tts.py              # TTS example
├── example_tts_turbo.py        # Turbo TTS example
├── example_vc.py               # Voice conversion example
├── example_for_mac.py          # Mac (MPS) example
├── gradio_tts_app.py           # Gradio TTS web UI
├── gradio_tts_turbo_app.py     # Gradio Turbo TTS web UI
├── gradio_vc_app.py            # Gradio voice conversion web UI
├── multilingual_app.py         # Multilingual TTS app
├── OPTIMIZATION_NOTES.md       # Detailed optimization notes
├── STREAMING_TODO.md           # Streaming feature tracker
└── pyproject.toml
```

## Key optimization files

- `OPTIMIZATION_NOTES.md` — detailed description of each optimization applied
- `STREAMING_TODO.md` — planned streaming improvements

## Running

```bash
pip install -e .
python example_tts_turbo.py
python gradio_tts_turbo_app.py
```

Requires PyTorch 2.6+, CUDA 12.4+, GCC.

## Upstream

https://github.com/resemble-ai/chatterbox — for model docs, supported languages, and evaluation reports.
