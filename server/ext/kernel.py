from torch.utils.cpp_extension import load
from pathlib import Path

FUSED_FLAGS = [
    "-O3",
    "-arch=native",
    "--use_fast_math",
    "--extra-device-vectorization",
    "-Xptxas",
    "-O3",
    "-Xptxas",
    "-v",
    "-Xptxas",
    "--allow-expensive-optimizations=true",
]


class FusedKernel:
    def __init__(self, kernel_name: str):
        self.kernel_name = kernel_name
        self._instances = {}

    def compute_hash_key(self, fused: bool, **kwargs):
        fused_str = "fused" if fused else "eager"
        return f"default_{fused_str}"

    def get_module(self, fused=True, **kwargs):
        hash_key = self.compute_hash_key(fused, **kwargs)
        if hash_key not in self._instances:
            print(f"Compiling kernel for config: {hash_key}...")
            ext_path = Path(__file__).parent / self.kernel_name
            module = load(
                name=f"{self.kernel_name}_{hash_key}",
                sources=[ext_path / "binding.cpp", ext_path / "kernel.cu"],
                extra_cuda_cflags=[
                    *[f"-D{k.upper()}={v}" for k, v in kwargs.items()],
                    *(FUSED_FLAGS if fused else []),
                ],
                verbose=True,
            )
            self._instances[hash_key] = module
        return self._instances[hash_key]


class DecodeNextTokenKernel(FusedKernel):
    def compute_hash_key(
        self,
        fused,
        model_attn_num_heads,
        model_hidden_dim,
        model_mlp_hidden_dim,
    ):
        fused_str = "fused" if fused else "eager"
        return f"default_{fused_str}_{model_attn_num_heads}_{model_hidden_dim}_{model_mlp_hidden_dim}"

    def __init__(self):
        super().__init__("decode_next_token")


class ProcessPromptKernel(FusedKernel):
    def compute_hash_key(
        self,
        fused,
        model_attn_num_heads,
        model_hidden_dim,
        model_mlp_hidden_dim,
        model_processing_seq_length,
    ):
        fused_str = "fused" if fused else "eager"
        return f"default_{fused_str}_{model_attn_num_heads}_{model_hidden_dim}_{model_mlp_hidden_dim}_{model_processing_seq_length}"

    def __init__(self):
        super().__init__("process_prompt")


process_prompt = ProcessPromptKernel()
decode_next_token = DecodeNextTokenKernel()
