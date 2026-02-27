# Thanks
Mostly from https://github.com/RVC-Boss/GPT-SoVITS with modifications.

# Usage
```bash
uv run --with huggingface-hub hf download neody/gpt-sovits-v2-pro-simple --local-dir ./data # Setup
uv run webui.py # Run webui, auto installs dependencies
```

# About
This is the current fastest implementation of gpt-sovits-v2-pro with not accuracy drop. Most of the optimization is by using torch.compile, removing v1/v3 features/dependencies that doesn't matter here. No quantization.

这是当前最快的 gpt-sovits-v2-pro 实现，并且精度没有下降。大多数优化是通过使用 torch.compile 进行的，并删除此处不重要的 v1/v3 功能/依赖项。没有量化。(Google translated sentence)

このレポジトリは間違いなく今あるgpt-sovtis-v2-proの実装の中で精度低下なしで最速のものです。最適化のほとんどはtorch.compileとv1/v3固有の実装や依存関係の削除によって行われています。量子化はしていません。
