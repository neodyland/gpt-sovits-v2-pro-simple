#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <math.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x. is_cuda(), #x " must be a CUDA tensor")
#define CHECK_HALF(x) TORCH_CHECK(x. scalar_type() == torch::kHalf, #x " must be float16")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
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
#ifndef MODEL_PROCESSING_SEQ_LENGTH
#error "MODEL_PROCESSING_SEQ_LENGTH must be defined"
#endif

static constexpr int H_DIM = MODEL_HIDDEN_DIM;
static constexpr int NUM_HEADS = MODEL_ATTN_NUM_HEADS;
static constexpr int HEAD_DIM = H_DIM / NUM_HEADS;
static constexpr int MLP_H_DIM = MODEL_MLP_HIDDEN_DIM;
static constexpr int BATCH_TOK = MODEL_PROCESSING_SEQ_LENGTH;
static constexpr int WARP_SIZE = 32;

struct TransformerBuffers
{
    half *qkv_batch;  // [BATCH_TOK, 3*H_DIM]
    half *attn_out;   // [BATCH_TOK, H_DIM]
    half *mlp_hidden; // [BATCH_TOK, MLP_H_DIM]
    half *residual;   // [BATCH_TOK, H_DIM]
    cudaStream_t stream;
    cublasHandle_t handle;
    bool initialized;
    int max_seq_len;
};

static thread_local TransformerBuffers g_buf = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, false, 0};

void ensure_buffers(int max_seq_len)
{
    if (!g_buf. initialized || max_seq_len > g_buf.max_seq_len)
    {
        if (g_buf. initialized)
        {
            cudaFree(g_buf.qkv_batch);
            cudaFree(g_buf. attn_out);
            cudaFree(g_buf.mlp_hidden);
            cudaFree(g_buf.residual);
        }
        else
        {
            cudaStreamCreate(&g_buf.stream);
            cublasCreate(&g_buf.handle);
            cublasSetMathMode(g_buf.handle, CUBLAS_TENSOR_OP_MATH);
        }

        cudaMalloc(&g_buf.qkv_batch, BATCH_TOK * 3 * H_DIM * sizeof(half));
        cudaMalloc(&g_buf.attn_out, BATCH_TOK * H_DIM * sizeof(half));
        cudaMalloc(&g_buf.mlp_hidden, BATCH_TOK * MLP_H_DIM * sizeof(half));
        cudaMalloc(&g_buf.residual, BATCH_TOK * H_DIM * sizeof(half));

        cublasSetStream(g_buf. handle, g_buf.stream);
        g_buf.max_seq_len = max_seq_len;
        g_buf.initialized = true;
    }
}

__global__ void batched_add_bias_kernel(half *__restrict__ output,
                                        const half *__restrict__ bias,
                                        int batch_size,
                                        int dim)
{
    int idx = blockIdx.x * blockDim. x + threadIdx. x;
    int total = batch_size * dim;

    if (idx < total)
    {
        int bias_idx = idx % dim;
        float val = __half2float(output[idx]) + __half2float(bias[bias_idx]);
        output[idx] = __float2half(val);
    }
}

__global__ void batched_qkv_bias_kv_update_kernel(
    half *__restrict__ qkv,        // [BATCH_TOK, 3*H_DIM]
    const half *__restrict__ bias, // [3*H_DIM]
    half *__restrict__ kv_cache,   // [2*max_len, H_DIM]
    int start_pos,
    int hidden_dim,
    int max_len,
    int batch_size)
{
    int idx = blockIdx. x * blockDim.x + threadIdx.x;
    int qkv_dim = 3 * hidden_dim;
    int total = batch_size * qkv_dim;

    if (idx < total)
    {
        int token_idx = idx / qkv_dim;
        int feat_idx = idx % qkv_dim;

        float val = __half2float(qkv[idx]) + __half2float(bias[feat_idx]);
        half h_val = __float2half(val);
        qkv[idx] = h_val;

        int pos = start_pos + token_idx;

        // Update K cache
        if (feat_idx >= hidden_dim && feat_idx < 2 * hidden_dim)
        {
            int k_idx = feat_idx - hidden_dim;
            kv_cache[pos * hidden_dim + k_idx] = h_val;
        }
        // Update V cache
        else if (feat_idx >= 2 * hidden_dim)
        {
            int v_idx = feat_idx - 2 * hidden_dim;
            kv_cache[(max_len + pos) * hidden_dim + v_idx] = h_val;
        }
    }
}

__global__ void batched_attention_kernel(
    const half *__restrict__ qkv,      // [BATCH_TOK, 3*H_DIM] - Q is first H_DIM
    const half *__restrict__ kv_cache, // [2*max_len, H_DIM]
    const bool *__restrict__ mask,     // [BATCH_TOK, max_seq_len]
    half *__restrict__ out,            // [BATCH_TOK, H_DIM]
    int start_pos,
    int hidden_dim,
    int max_len,
    int mask_stride,
    int batch_size)
{
    // Block (token_idx, head_idx)
    int token_idx = blockIdx. x;
    int head_idx = blockIdx. y;
    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    if (token_idx >= batch_size)
        return;

    int seq_len = start_pos + token_idx + 1;
    const int num_warps = (num_threads + WARP_SIZE - 1) / WARP_SIZE;

    extern __shared__ float smem[];
    float *scores = smem;
    float *reduce_buf = scores + seq_len;
    // Use one extra location to broadcast final max and sum
    float *broadcast_val = reduce_buf + num_warps;

    // Q for this token and head
    const half *q_head = qkv + token_idx * 3 * hidden_dim + head_idx * HEAD_DIM;
    const float scale = rsqrtf((float)HEAD_DIM);

    // Compute attention scores
    float local_max = -1e20f;
    const bool *token_mask = mask + token_idx * mask_stride;

    for (int j = tid; j < seq_len; j += num_threads)
    {
        const half *k_head = kv_cache + j * hidden_dim + head_idx * HEAD_DIM;
        float dot = 0.0f;

#pragma unroll 8
        for (int d = 0; d < HEAD_DIM; d++)
        {
            dot += __half2float(q_head[d]) * __half2float(k_head[d]);
        }
        dot *= scale;

        if (token_mask && ! token_mask[j])
            dot = -1e9f;

        scores[j] = dot;
        local_max = fmaxf(local_max, dot);
    }
    __syncthreads();

    // Warp reduce max
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));

    int lane = tid % WARP_SIZE;
    int wid = tid / WARP_SIZE;
    if (lane == 0)
        reduce_buf[wid] = local_max;
    __syncthreads();

    // Final reduction in first warp and broadcast via shared memory
    if (tid < num_warps)
        local_max = reduce_buf[tid];
    else
        local_max = -1e20f;
    
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
    
    // Thread 0 writes the final max to shared memory for all threads
    if (tid == 0)
        broadcast_val[0] = local_max;
    __syncthreads();
    
    float max_val = broadcast_val[0];

    // Softmax exp and sum
    float local_sum = 0.0f;
    for (int j = tid; j < seq_len; j += num_threads)
    {
        float e = expf(scores[j] - max_val);
        scores[j] = e;
        local_sum += e;
    }
    __syncthreads();

    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);

    if (lane == 0)
        reduce_buf[wid] = local_sum;
    __syncthreads();

    // Final reduction in first warp and broadcast via shared memory
    if (tid < num_warps)
        local_sum = reduce_buf[tid];
    else
        local_sum = 0.0f;
    
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);

    // Thread 0 writes the final sum to shared memory for all threads
    if (tid == 0)
        broadcast_val[0] = local_sum;
    __syncthreads();

    float sum_val = broadcast_val[0];
    float inv_sum = 1.0f / (sum_val + 1e-9f);

    // Weighted sum of values
    half *out_head = out + token_idx * hidden_dim + head_idx * HEAD_DIM;
    for (int d = tid; d < HEAD_DIM; d += num_threads)
    {
        float acc = 0.0f;
        for (int j = 0; j < seq_len; j++)
        {
            const half *v_head = kv_cache + (max_len + j) * hidden_dim + head_idx * HEAD_DIM;
            acc += scores[j] * __half2float(v_head[d]);
        }
        out_head[d] = __float2half(acc * inv_sum);
    }
}

__global__ void batched_bias_relu_kernel(half *__restrict__ data,
                                         const half *__restrict__ bias,
                                         int batch_size,
                                         int dim)
{
    int idx = blockIdx.x * blockDim. x + threadIdx. x;
    int total = batch_size * dim;

    if (idx < total)
    {
        int bias_idx = idx % dim;
        float val = __half2float(data[idx]) + __half2float(bias[bias_idx]);
        data[idx] = __float2half(fmaxf(0.0f, val));
    }
}

__global__ void batched_bias_residual_kernel(half *__restrict__ out,
                                             const half *__restrict__ bias,
                                             const half *__restrict__ residual,
                                             int batch_size,
                                             int dim)
{
    int idx = blockIdx.x * blockDim. x + threadIdx. x;
    int total = batch_size * dim;

    if (idx < total)
    {
        int bias_idx = idx % dim;
        float val = __half2float(out[idx]) + __half2float(bias[bias_idx]) + __half2float(residual[idx]);
        out[idx] = __float2half(val);
    }
}

__global__ void batched_layernorm_kernel(half *__restrict__ x,
                                         const half *__restrict__ gamma,
                                         const half *__restrict__ beta,
                                         int hidden_dim,
                                         float eps)
{
    int token_idx = blockIdx.x;
    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    half *x_row = x + token_idx * hidden_dim;

    extern __shared__ float smem[];
    float *reduce_buf = smem;

    // Compute mean
    float local_sum = 0.0f;
    for (int i = tid; i < hidden_dim; i += num_threads)
        local_sum += __half2float(x_row[i]);

    reduce_buf[tid] = local_sum;
    __syncthreads();

    for (int stride = num_threads / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
            reduce_buf[tid] += reduce_buf[tid + stride];
        __syncthreads();
    }

    float mean = reduce_buf[0] / hidden_dim;
    __syncthreads();

    // Compute variance
    float local_var = 0.0f;
    for (int i = tid; i < hidden_dim; i += num_threads)
    {
        float diff = __half2float(x_row[i]) - mean;
        local_var += diff * diff;
    }

    reduce_buf[tid] = local_var;
    __syncthreads();

    for (int stride = num_threads / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
            reduce_buf[tid] += reduce_buf[tid + stride];
        __syncthreads();
    }

    float var = reduce_buf[0] / hidden_dim;
    float inv_std = rsqrtf(var + eps);

    for (int i = tid; i < hidden_dim; i += num_threads)
    {
        float val = __half2float(x_row[i]);
        float norm = (val - mean) * inv_std;
        x_row[i] = __float2half(norm * __half2float(gamma[i]) + __half2float(beta[i]));
    }
}

__global__ void batched_fused_add_ln_kernel(const half *__restrict__ residual,
                                            half *__restrict__ x,
                                            half *__restrict__ x_copy,
                                            const half *__restrict__ gamma,
                                            const half *__restrict__ beta,
                                            int hidden_dim,
                                            float eps)
{
    int token_idx = blockIdx.x;
    int tid = threadIdx.x;
    int num_threads = blockDim. x;

    const half *res_row = residual + token_idx * hidden_dim;
    half *x_row = x + token_idx * hidden_dim;
    half *copy_row = x_copy ?  x_copy + token_idx * hidden_dim : nullptr;

    extern __shared__ char smem_raw[];
    float *temp = reinterpret_cast<float *>(smem_raw);
    float *reduce_buf = temp + hidden_dim;

    float local_sum = 0.0f;
    for (int i = tid; i < hidden_dim; i += num_threads)
    {
        float val = __half2float(x_row[i]) + __half2float(res_row[i]);
        temp[i] = val;
        local_sum += val;
    }

    reduce_buf[tid] = local_sum;
    __syncthreads();

    for (int stride = num_threads / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
            reduce_buf[tid] += reduce_buf[tid + stride];
        __syncthreads();
    }

    float mean = reduce_buf[0] / hidden_dim;
    __syncthreads();

    float local_var = 0.0f;
    for (int i = tid; i < hidden_dim; i += num_threads)
    {
        float diff = temp[i] - mean;
        local_var += diff * diff;
    }

    reduce_buf[tid] = local_var;
    __syncthreads();

    for (int stride = num_threads / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
            reduce_buf[tid] += reduce_buf[tid + stride];
        __syncthreads();
    }

    float var = reduce_buf[0] / hidden_dim;
    float inv_std = rsqrtf(var + eps);

    for (int i = tid; i < hidden_dim; i += num_threads)
    {
        float norm = (temp[i] - mean) * inv_std;
        half result = __float2half(norm * __half2float(gamma[i]) + __half2float(beta[i]));
        x_row[i] = result;
        if (copy_row)
            copy_row[i] = result;
    }
}

void forward_cuda_batch(
    torch::Tensor x_batch,
    torch:: Tensor kv_cache,
    int64_t idx,
    torch:: Tensor attn_mask,
    torch:: Tensor w_qkv, torch::Tensor b_qkv,
    torch:: Tensor w_out, torch::Tensor b_out,
    torch::Tensor gamma1, torch::Tensor beta1,
    torch::Tensor gamma2, torch::Tensor beta2,
    torch::Tensor mlp_w1, torch::Tensor mlp_b1,
    torch::Tensor mlp_w2, torch::Tensor mlp_b2)
{
    CHECK_INPUT(x_batch);
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
    CHECK_CUDA(attn_mask);

    const int max_len = kv_cache.size(1);
    const int qkv_dim = 3 * H_DIM;
    const int threads = 256;
    const int num_warps = (threads + WARP_SIZE - 1) / WARP_SIZE;

    ensure_buffers(max_len);

    cudaStream_t stream = g_buf.stream;
    cublasHandle_t handle = g_buf.handle;

    const __half alpha_h = __float2half(1.0f);
    const __half beta_h = __float2half(0.0f);

    half *x_ptr = reinterpret_cast<half *>(x_batch.data_ptr<at::Half>());
    half *kv_ptr = reinterpret_cast<half *>(kv_cache.data_ptr<at::Half>());
    half *w_qkv_ptr = reinterpret_cast<half *>(w_qkv.data_ptr<at::Half>());
    half *b_qkv_ptr = reinterpret_cast<half *>(b_qkv.data_ptr<at::Half>());
    half *w_out_ptr = reinterpret_cast<half *>(w_out.data_ptr<at::Half>());
    half *b_out_ptr = reinterpret_cast<half *>(b_out.data_ptr<at:: Half>());
    half *gamma1_ptr = reinterpret_cast<half *>(gamma1.data_ptr<at::Half>());
    half *beta1_ptr = reinterpret_cast<half *>(beta1.data_ptr<at::Half>());
    half *gamma2_ptr = reinterpret_cast<half *>(gamma2.data_ptr<at::Half>());
    half *beta2_ptr = reinterpret_cast<half *>(beta2.data_ptr<at:: Half>());
    half *mlp_w1_ptr = reinterpret_cast<half *>(mlp_w1.data_ptr<at::Half>());
    half *mlp_b1_ptr = reinterpret_cast<half *>(mlp_b1.data_ptr<at::Half>());
    half *mlp_w2_ptr = reinterpret_cast<half *>(mlp_w2.data_ptr<at::Half>());
    half *mlp_b2_ptr = reinterpret_cast<half *>(mlp_b2.data_ptr<at::Half>());
    bool *mask_ptr = reinterpret_cast<bool *>(attn_mask.data_ptr<bool>());

    int mask_stride = attn_mask.size(1);

    CUBLAS_CHECK(cublasHgemm(handle,
                             CUBLAS_OP_T, CUBLAS_OP_N,
                             qkv_dim, BATCH_TOK, H_DIM,
                             &alpha_h,
                             w_qkv_ptr, H_DIM,
                             x_ptr, H_DIM,
                             &beta_h,
                             g_buf.qkv_batch, qkv_dim));

    int total_qkv = BATCH_TOK * qkv_dim;
    int blocks = (total_qkv + threads - 1) / threads;
    batched_qkv_bias_kv_update_kernel<<<blocks, threads, 0, stream>>>(
        g_buf.qkv_batch, b_qkv_ptr, kv_ptr, idx, H_DIM, max_len, BATCH_TOK);

    int max_seq = idx + BATCH_TOK;
    // Shared memory:  scores[max_seq] + reduce_buf[num_warps] + broadcast_val[1]
    size_t attn_smem = (max_seq + num_warps + 1) * sizeof(float);
    dim3 attn_grid(BATCH_TOK, NUM_HEADS);
    batched_attention_kernel<<<attn_grid, threads, attn_smem, stream>>>(
        g_buf.qkv_batch, kv_ptr, mask_ptr, g_buf.attn_out,
        idx, H_DIM, max_len, mask_stride, BATCH_TOK);

    CUBLAS_CHECK(cublasHgemm(handle,
                             CUBLAS_OP_T, CUBLAS_OP_N,
                             H_DIM, BATCH_TOK, H_DIM,
                             &alpha_h,
                             w_out_ptr, H_DIM,
                             g_buf.attn_out, H_DIM,
                             &beta_h,
                             g_buf.attn_out, H_DIM));

    int total_h = BATCH_TOK * H_DIM;
    blocks = (total_h + threads - 1) / threads;
    batched_add_bias_kernel<<<blocks, threads, 0, stream>>>(
        g_buf.attn_out, b_out_ptr, BATCH_TOK, H_DIM);

    size_t ln_smem = (H_DIM + threads) * sizeof(float);
    batched_fused_add_ln_kernel<<<BATCH_TOK, threads, ln_smem, stream>>>(
        x_ptr, g_buf.attn_out, g_buf.residual, gamma1_ptr, beta1_ptr, H_DIM, 1e-5f);

    cudaMemcpyAsync(x_ptr, g_buf.attn_out, BATCH_TOK * H_DIM * sizeof(half),
                    cudaMemcpyDeviceToDevice, stream);
    CUBLAS_CHECK(cublasHgemm(handle,
                             CUBLAS_OP_T, CUBLAS_OP_N,
                             MLP_H_DIM, BATCH_TOK, H_DIM,
                             &alpha_h,
                             mlp_w1_ptr, H_DIM,
                             g_buf.attn_out, H_DIM,
                             &beta_h,
                             g_buf.mlp_hidden, MLP_H_DIM));
    int total_mlp = BATCH_TOK * MLP_H_DIM;
    blocks = (total_mlp + threads - 1) / threads;
    batched_bias_relu_kernel<<<blocks, threads, 0, stream>>>(
        g_buf.mlp_hidden, mlp_b1_ptr, BATCH_TOK, MLP_H_DIM);
    CUBLAS_CHECK(cublasHgemm(handle,
                             CUBLAS_OP_T, CUBLAS_OP_N,
                             H_DIM, BATCH_TOK, MLP_H_DIM,
                             &alpha_h,
                             mlp_w2_ptr, MLP_H_DIM,
                             g_buf.mlp_hidden, MLP_H_DIM,
                             &beta_h,
                             g_buf.attn_out, H_DIM));
    blocks = (total_h + threads - 1) / threads;
    batched_bias_residual_kernel<<<blocks, threads, 0, stream>>>(
        g_buf.attn_out, mlp_b2_ptr, g_buf.residual, BATCH_TOK, H_DIM);
    size_t ln2_smem = threads * sizeof(float);
    batched_layernorm_kernel<<<BATCH_TOK, threads, ln2_smem, stream>>>(
        g_buf.attn_out, gamma2_ptr, beta2_ptr, H_DIM, 1e-5f);
    cudaMemcpyAsync(x_ptr, g_buf.attn_out, BATCH_TOK * H_DIM * sizeof(half),
                    cudaMemcpyDeviceToDevice, stream);

    cudaStreamSynchronize(stream);
}