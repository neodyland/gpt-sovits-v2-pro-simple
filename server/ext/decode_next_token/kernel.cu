#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <math.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_HALF(x) TORCH_CHECK(x.scalar_type() == torch::kHalf, #x " must be float16")
#define CHECK_INPUT(x)   \
    CHECK_CUDA(x);       \
    CHECK_HALF(x)

#define CUBLAS_CHECK(err)                             \
    do                                                \
    {                                                 \
        if (err != CUBLAS_STATUS_SUCCESS)             \
        {                                             \
            throw std::runtime_error("cuBLAS error"); \
        }                                             \
    } while (0)

#ifndef MODEL_ATTN_NUM_HEADS
#error "MODEL_ATTN_NUM_HEADS must be defined"
#endif
#ifndef MODEL_HIDDEN_DIM
#error "MODEL_HIDDEN_DIM must be defined"
#endif
#ifndef MODEL_MLP_HIDDEN_DIM
#error "MODEL_MLP_HIDDEN_DIM must be defined"
#endif

static constexpr int H_DIM = MODEL_HIDDEN_DIM;
static constexpr int NUM_HEADS = MODEL_ATTN_NUM_HEADS;
static constexpr int HEAD_DIM = H_DIM / NUM_HEADS;
static constexpr int MLP_H_DIM = MODEL_MLP_HIDDEN_DIM;

static thread_local cublasHandle_t cublas_handle = nullptr;
static __half *g_mlp_hidden = nullptr;
static int g_mlp_hidden_capacity = 0;

cublasHandle_t get_cublas_handle()
{
    if (cublas_handle == nullptr)
    {
        CUBLAS_CHECK(cublasCreate(&cublas_handle));
        CUBLAS_CHECK(cublasSetMathMode(cublas_handle, CUBLAS_TENSOR_OP_MATH));
    }
    return cublas_handle;
}

__global__ void add_bias_kernel(
    half *__restrict__ output,
    const half *__restrict__ bias,
    int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        float val = __half2float(output[idx]) + __half2float(bias[idx]);
        output[idx] = __float2half(val);
    }
}

__global__ void bias_relu_kernel(__half *data, const __half *bias, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        float v = __half2float(data[i]) + __half2float(bias[i]);
        data[i] = __float2half(fmaxf(0.0f, v));
    }
}

__global__ void bias_residual_kernel(
    __half *__restrict__ out,
    const __half *__restrict__ bias,
    const __half *__restrict__ residual,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        float v = __half2float(out[i]) + __half2float(bias[i]) + __half2float(residual[i]);
        out[i] = __float2half(v);
    }
}

__global__ void update_kv_cache_kernel(
    const half *__restrict__ kv,
    half *__restrict__ kv_cache,
    int pos,
    int hidden_dim,
    int max_len)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < hidden_dim)
    {
        kv_cache[pos * hidden_dim + idx] = kv[idx];
        kv_cache[(max_len + pos) * hidden_dim + idx] = kv[hidden_dim + idx];
    }
}

__global__ void attention_kernel_optimized(
    const half *__restrict__ q,
    const half *__restrict__ kv_cache,
    half *__restrict__ out,
    int seq_len,
    int hidden_dim,
    int max_len)
{
    const int h = blockIdx.x;
    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;

    extern __shared__ float shared_mem[];
    float *scores = shared_mem;
    float *reduce_buf = shared_mem + seq_len;

    const half *q_head = q + h * HEAD_DIM;
    const float scale = rsqrtf((float)HEAD_DIM);

    float local_max = -1e20f;
    for (int j = tid; j < seq_len; j += num_threads)
    {
        const half *k_head = kv_cache + j * hidden_dim + h * HEAD_DIM;
        float dot = 0.0f;

        for (int d = 0; d < HEAD_DIM; d++)
        {
            dot += __half2float(q_head[d]) * __half2float(k_head[d]);
        }

        dot *= scale;
        scores[j] = dot;
        local_max = fmaxf(local_max, dot);
    }

    reduce_buf[tid] = local_max;
    __syncthreads();

    for (int stride = num_threads / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride && tid + stride < num_threads)
        {
            reduce_buf[tid] = fmaxf(reduce_buf[tid], reduce_buf[tid + stride]);
        }
        __syncthreads();
    }
    float max_val = reduce_buf[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (int j = tid; j < seq_len; j += num_threads)
    {
        float e = expf(scores[j] - max_val);
        scores[j] = e;
        local_sum += e;
    }

    reduce_buf[tid] = local_sum;
    __syncthreads();

    for (int stride = num_threads / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride && tid + stride < num_threads)
        {
            reduce_buf[tid] += reduce_buf[tid + stride];
        }
        __syncthreads();
    }
    float sum_val = reduce_buf[0];
    float inv_sum = 1.0f / (sum_val + 1e-9f);
    __syncthreads();

    half *out_head = out + h * HEAD_DIM;
    for (int d = tid; d < HEAD_DIM; d += num_threads)
    {
        float acc = 0.0f;
        for (int j = 0; j < seq_len; j++)
        {
            const half *v_head = kv_cache + (max_len + j) * hidden_dim + h * HEAD_DIM;
            acc += scores[j] * __half2float(v_head[d]);
        }
        out_head[d] = __float2half(acc * inv_sum);
    }
}

// Fused residual + layernorm
__global__ void fused_add_layernorm_kernel(
    const half *__restrict__ residual,
    half *__restrict__ x,
    const half *__restrict__ gamma,
    const half *__restrict__ beta,
    int hidden_dim,
    float eps)
{
    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;

    extern __shared__ float shmem[];
    float *temp = shmem;
    float *reduce_buf = shmem + hidden_dim;

    float local_sum = 0.0f;
    float local_sumsq = 0.0f;

    for (int i = tid; i < hidden_dim; i += num_threads)
    {
        float v = __half2float(x[i]) + __half2float(residual[i]);
        temp[i] = v;
        local_sum += v;
        local_sumsq += v * v;
    }

    reduce_buf[tid] = local_sum;
    reduce_buf[tid + num_threads] = local_sumsq;
    __syncthreads();

    for (int stride = num_threads / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            reduce_buf[tid] += reduce_buf[tid + stride];
            reduce_buf[tid + num_threads] += reduce_buf[tid + num_threads + stride];
        }
        __syncthreads();
    }

    float mean = reduce_buf[0] / hidden_dim;
    float var = (reduce_buf[num_threads] / hidden_dim) - (mean * mean);
    float inv_std = rsqrtf(var + eps);

    for (int i = tid; i < hidden_dim; i += num_threads)
    {
        float norm = (temp[i] - mean) * inv_std;
        float g = __half2float(gamma[i]);
        float b = __half2float(beta[i]);
        x[i] = __float2half(norm * g + b);
    }
}

// Layernorm kernel
__global__ void layernorm_kernel(
    half *__restrict__ x,
    const half *__restrict__ gamma,
    const half *__restrict__ beta,
    int hidden_dim,
    float eps)
{
    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;

    extern __shared__ float shmem[];
    float *reduce_buf = shmem;

    float local_sum = 0.0f;
    float local_sumsq = 0.0f;

    for (int i = tid; i < hidden_dim; i += num_threads)
    {
        float v = __half2float(x[i]);
        local_sum += v;
        local_sumsq += v * v;
    }

    reduce_buf[tid] = local_sum;
    reduce_buf[tid + num_threads] = local_sumsq;
    __syncthreads();

    for (int stride = num_threads / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            reduce_buf[tid] += reduce_buf[tid + stride];
            reduce_buf[tid + num_threads] += reduce_buf[tid + num_threads + stride];
        }
        __syncthreads();
    }

    float mean = reduce_buf[0] / hidden_dim;
    float var = (reduce_buf[num_threads] / hidden_dim) - (mean * mean);
    float inv_std = rsqrtf(var + eps);
    __syncthreads();

    for (int i = tid; i < hidden_dim; i += num_threads)
    {
        float v = __half2float(x[i]);
        float norm = (v - mean) * inv_std;
        float g = __half2float(gamma[i]);
        float b = __half2float(beta[i]);
        x[i] = __float2half(norm * g + b);
    }
}

void forward_cuda(
    torch::Tensor x, torch::Tensor kv_cache, int64_t idx,
    torch::Tensor w_qkv, torch::Tensor b_qkv,
    torch::Tensor w_out, torch::Tensor b_out,
    torch::Tensor gamma1, torch::Tensor beta1,
    torch::Tensor gamma2, torch::Tensor beta2,
    torch::Tensor mlp_w1, torch::Tensor mlp_b1,
    torch::Tensor mlp_w2, torch::Tensor mlp_b2)
{
    CHECK_INPUT(x);
    CHECK_INPUT(kv_cache);
    CHECK_INPUT(w_qkv);
    CHECK_INPUT(b_qkv);
    CHECK_INPUT(w_out);
    CHECK_INPUT(b_out);
    CHECK_INPUT(gamma1);
    CHECK_INPUT(beta1);
    CHECK_INPUT(gamma2);
    CHECK_INPUT(beta2);
    CHECK_INPUT(mlp_w1);
    CHECK_INPUT(mlp_b1);
    CHECK_INPUT(mlp_w2);
    CHECK_INPUT(mlp_b2);

    cublasHandle_t handle = get_cublas_handle();
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cublasSetStream(handle, stream);

    const int max_len = kv_cache.size(1);
    const int qkv_dim = w_qkv.size(0);
    const int pos = idx + 1;
    const int seq_len = pos;

    if (MLP_H_DIM > g_mlp_hidden_capacity)
    {
        if (g_mlp_hidden)
            cudaFree(g_mlp_hidden);
        cudaMalloc(&g_mlp_hidden, MLP_H_DIM * sizeof(__half));
        g_mlp_hidden_capacity = MLP_H_DIM;
    }

    auto options = x.options();
    auto qkv = torch::empty({qkv_dim}, options);
    auto state = torch::empty({H_DIM}, options);
    auto mlp_input = torch::empty({H_DIM}, options);

    half *x_ptr = reinterpret_cast<half *>(x.data_ptr<at::Half>());
    half *w_qkv_ptr = reinterpret_cast<half *>(w_qkv.data_ptr<at::Half>());
    half *b_qkv_ptr = reinterpret_cast<half *>(b_qkv.data_ptr<at::Half>());
    half *qkv_ptr = reinterpret_cast<half *>(qkv.data_ptr<at::Half>());
    half *w_out_ptr = reinterpret_cast<half *>(w_out.data_ptr<at::Half>());
    half *b_out_ptr = reinterpret_cast<half *>(b_out.data_ptr<at::Half>());
    half *kv_cache_ptr = reinterpret_cast<half *>(kv_cache.data_ptr<at::Half>());
    half *gamma1_ptr = reinterpret_cast<half *>(gamma1.data_ptr<at::Half>());
    half *beta1_ptr = reinterpret_cast<half *>(beta1.data_ptr<at::Half>());
    half *gamma2_ptr = reinterpret_cast<half *>(gamma2.data_ptr<at::Half>());
    half *beta2_ptr = reinterpret_cast<half *>(beta2.data_ptr<at::Half>());
    half *state_ptr = reinterpret_cast<half *>(state.data_ptr<at::Half>());
    half *mlp_input_ptr = reinterpret_cast<half *>(mlp_input.data_ptr<at::Half>());

    half *mlp_w1_ptr = reinterpret_cast<half *>(mlp_w1.data_ptr<at::Half>());
    half *mlp_b1_ptr = reinterpret_cast<half *>(mlp_b1.data_ptr<at::Half>());
    half *mlp_w2_ptr = reinterpret_cast<half *>(mlp_w2.data_ptr<at::Half>());
    half *mlp_b2_ptr = reinterpret_cast<half *>(mlp_b2.data_ptr<at::Half>());

    const int threads = 256;

    const __half alpha = __float2half(1.0f);
    const __half beta_zero = __float2half(0.0f);

    CUBLAS_CHECK(cublasHgemm(handle,
                             CUBLAS_OP_T, CUBLAS_OP_N,
                             qkv_dim, 1, H_DIM,
                             &alpha,
                             w_qkv_ptr, H_DIM,
                             x_ptr, H_DIM,
                             &beta_zero,
                             qkv_ptr, qkv_dim));

    int blocks = (qkv_dim + threads - 1) / threads;
    add_bias_kernel<<<blocks, threads, 0, stream>>>(qkv_ptr, b_qkv_ptr, qkv_dim);

    blocks = (H_DIM + threads - 1) / threads;
    update_kv_cache_kernel<<<blocks, threads, 0, stream>>>(
        qkv_ptr + H_DIM, kv_cache_ptr, pos, H_DIM, max_len);

    size_t attn_shared = (seq_len + threads) * sizeof(float);
    attention_kernel_optimized<<<NUM_HEADS, threads, attn_shared, stream>>>(
        qkv_ptr, kv_cache_ptr, state_ptr, seq_len, H_DIM, max_len);

    CUBLAS_CHECK(cublasHgemm(handle,
                             CUBLAS_OP_T, CUBLAS_OP_N,
                             H_DIM, 1, H_DIM,
                             &alpha,
                             w_out_ptr, H_DIM,
                             state_ptr, H_DIM,
                             &beta_zero,
                             state_ptr, H_DIM));

    blocks = (H_DIM + threads - 1) / threads;
    add_bias_kernel<<<blocks, threads, 0, stream>>>(state_ptr, b_out_ptr, H_DIM);

    size_t ln_shared = (H_DIM + threads * 2) * sizeof(float);
    fused_add_layernorm_kernel<<<1, threads, ln_shared, stream>>>(
        x_ptr, state_ptr, gamma1_ptr, beta1_ptr, H_DIM, 1e-5f);

    cudaMemcpyAsync(x_ptr, state_ptr, H_DIM * sizeof(half), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(mlp_input_ptr, state_ptr, H_DIM * sizeof(half), cudaMemcpyDeviceToDevice, stream);

    CUBLAS_CHECK(cublasHgemm(handle,
                             CUBLAS_OP_T, CUBLAS_OP_N,
                             MLP_H_DIM, 1, H_DIM,
                             &alpha,
                             mlp_w1_ptr, H_DIM,
                             x_ptr, H_DIM,
                             &beta_zero,
                             g_mlp_hidden, MLP_H_DIM));

    blocks = (MLP_H_DIM + threads - 1) / threads;
    bias_relu_kernel<<<blocks, threads, 0, stream>>>(g_mlp_hidden, mlp_b1_ptr, MLP_H_DIM);

    CUBLAS_CHECK(cublasHgemm(handle,
                             CUBLAS_OP_T, CUBLAS_OP_N,
                             H_DIM, 1, MLP_H_DIM,
                             &alpha,
                             mlp_w2_ptr, MLP_H_DIM,
                             g_mlp_hidden, MLP_H_DIM,
                             &beta_zero,
                             state_ptr, H_DIM));

    blocks = (H_DIM + threads - 1) / threads;
    bias_residual_kernel<<<blocks, threads, 0, stream>>>(
        state_ptr, mlp_b2_ptr, mlp_input_ptr, H_DIM);

    size_t ln2_shared = threads * 2 * sizeof(float);
    layernorm_kernel<<<1, threads, ln2_shared, stream>>>(
        state_ptr, gamma2_ptr, beta2_ptr, H_DIM, 1e-5f);

    cudaMemcpyAsync(x_ptr, state_ptr, H_DIM * sizeof(half), cudaMemcpyDeviceToDevice, stream);
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
}