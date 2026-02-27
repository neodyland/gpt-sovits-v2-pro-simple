import torch
from .kernel import decode_next_token
from ..models import T2S
from ..ar.t2s_model import TS2Cache
import json
import torch._inductor.config as inductor_config

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model = T2S(device, dtype)
block = model.model.blocks[0]
cfg = json.load(open("./data/gsv/config.json", "r"))

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("medium")
torch.autograd.set_grad_enabled(False)
inductor_config.freezing = True
inductor_config.epilogue_fusion = True


def benchmark(label, func, iterations=1000, warmup=100):
    for _ in range(warmup):
        func()
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(iterations):
        func()
    end_event.record()
    torch.cuda.synchronize()
    avg_ms = start_event.elapsed_time(end_event) / iterations
    print(f"[{label}] AVG: {avg_ms:.4f} ms")
    return avg_ms


if __name__ == "__main__":
    x = torch.rand(
        1, 1, cfg["model"]["hidden_dim"], dtype=dtype, device=device
    ).contiguous()
    kernel = decode_next_token.get_module(
        fused=True,
        model_attn_num_heads=cfg["model"]["head"],
        model_hidden_dim=cfg["model"]["hidden_dim"],
        model_mlp_hidden_dim=cfg["model"]["hidden_dim"] * 4,
    )
    w1 = block.mlp.linear1.weight.clone().contiguous()
    b1 = block.mlp.linear1.bias.clone().contiguous()
    w2 = block.mlp.linear2.weight.clone().contiguous()
    b2 = block.mlp.linear2.bias.clone().contiguous()
    w_qkv = block.qkv.weight.clone().contiguous()
    b_qkv = block.qkv.bias.clone().contiguous()
    w_out = block.out_proj.weight.clone().contiguous()
    b_out = block.out_proj.bias.clone().contiguous()
    gamma1 = block.norm_1.weight.clone().contiguous()
    beta1 = block.norm_1.bias.clone().contiguous()
    gamma2 = block.norm_2.weight.clone().contiguous()
    beta2 = block.norm_2.bias.clone().contiguous()
    cache = TS2Cache(2048, cfg["model"]["hidden_dim"], "cuda", torch.float16)
    kv_cache = cache.kv_cache.clone().contiguous()
    idx = int(0)

    def run_custom():
        d = kernel.forward(
            x.clone().contiguous(),
            kv_cache,
            idx,
            w_qkv,
            b_qkv,
            w_out,
            b_out,
            gamma1,
            beta1,
            gamma2,
            beta2,
            w1,
            b1,
            w2,
            b2,
        ).float()
        return d

    benchmark("Custom Fused Kernel", run_custom)

    forward = torch.compile(block.decode_next_token_raw, fullgraph=True, dynamic=True)

    @torch.inference_mode()
    def run_pytorch():
        return forward(x, cache, idx).float()

    for i in range(10):
        run_pytorch()

    benchmark("PyTorch Original MLP", run_pytorch)
    out_c = run_custom()
    out_t = run_pytorch()
    diff = (out_c - out_t).abs()

    print("max abs:", diff.max().item())
    print("mean abs:", diff.mean().item())
    print(
        "custom stats:",
        out_c.min().item(),
        out_c.max().item(),
        out_c.mean().item(),
        out_c.std().item(),
    )
    print(
        "torch stats:",
        out_t.min().item(),
        out_t.max().item(),
        out_t.mean().item(),
        out_t.std().item(),
    )
    print(
        torch.allclose(
            run_custom(),
            run_pytorch(),
            rtol=1e-3,  # relative tolerance
            atol=1e-3,
        )
    )
    print(cache.kv_cache, kv_cache)
