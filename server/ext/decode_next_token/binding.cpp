#include <torch/extension.h>
#include <vector>

void forward_cuda(
    torch::Tensor x,
    torch::Tensor kv_cache,
    int64_t idx,
    torch::Tensor w_qkv, torch::Tensor b_qkv,
    torch::Tensor w_out, torch::Tensor b_out,
    torch::Tensor gamma1, torch::Tensor beta1,
    torch::Tensor gamma2, torch::Tensor beta2,
    torch::Tensor mlp_w1, torch::Tensor mlp_b1,
    torch::Tensor mlp_w2, torch::Tensor mlp_b2);

torch::Tensor forward_bridge(
    torch::Tensor x,
    torch::Tensor kv_cache,
    int64_t idx,
    torch::Tensor w_qkv, torch::Tensor b_qkv,
    torch::Tensor w_out, torch::Tensor b_out,
    torch::Tensor gamma1, torch::Tensor beta1,
    torch::Tensor gamma2, torch::Tensor beta2,
    torch::Tensor mlp_w1, torch::Tensor mlp_b1,
    torch::Tensor mlp_w2, torch::Tensor mlp_b2)
{
    forward_cuda(
        x,
        kv_cache,
        idx,
        w_qkv, b_qkv,
        w_out, b_out,
        gamma1, beta1,
        gamma2, beta2,
        mlp_w1, mlp_b1,
        mlp_w2, mlp_b2);
    return x;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &forward_bridge, "Fused kernel");
}