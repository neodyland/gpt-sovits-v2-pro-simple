# Thanks
Mostly from https://github.com/RVC-Boss/GPT-SoVITS with modifications.

# Usage
```bash
uv run --with huggingface-hub hf download neody/gpt-sovits-v2-pro-simple --local-dir ./data # Setup
uv run webui.py # Run webui, auto installs dependencies
```

## Convert tips
We don't need `PosteriorEncoder`(`enc_q`) as its not used in inference.