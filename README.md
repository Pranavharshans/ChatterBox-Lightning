# ChatterBox-Lightning

Drop-in optimized fork of [Chatterbox TTS](https://github.com/resemble-ai/chatterbox) by Resemble AI. **30-50% faster inference with zero configuration and identical audio quality.**

Optimizations apply automatically when the model loads — no code changes needed.

## Benchmarks (Turbo Model, 7 prompts average)

| GPU | Original | Optimized | RTF | Improvement |
|-----|----------|-----------|-----|-------------|
| RTX 3060 (12GB) | 4.22s | **2.93s** | 0.279 | **-30.6%** |
| RTX 4070 (12GB) | 5.33s | **2.65s** | 0.277 | **-50.3%** |
| RTX 4060 Ti (16GB) | 3.90s | **2.21s** | 0.231 | **-43.3%** |

## How It Works

Pre-allocated KV cache + BF16 Flash SDPA + `torch.compile` on the GPT-2 backbone. No model weights are changed — audio output is identical to upstream.

## Requirements

- PyTorch 2.6+
- CUDA 12.4+
- GCC

## Installation

```bash
git clone https://github.com/Pranavharshans/ChatterBox-Lightning.git
cd ChatterBox-Lightning
pip install -e .
```

## Usage

```python
from chatterbox.tts_turbo import ChatterboxTurboTTS

model = ChatterboxTurboTTS.from_pretrained(device="cuda")  # auto-optimized!
wav = model.generate("Hello world!")
```

See `example_tts.py`, `example_tts_turbo.py`, `example_vc.py`, and `example_for_mac.py` for more examples. Gradio apps are available in `gradio_tts_app.py` and `gradio_tts_turbo_app.py`.

## Upstream

This is a fork of [resemble-ai/chatterbox](https://github.com/resemble-ai/chatterbox). For model documentation, supported languages, evaluation reports, and the model zoo, see the [upstream README](https://github.com/resemble-ai/chatterbox).

## License

MIT — same as upstream.
