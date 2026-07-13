#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be CUDA")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_INT32(x) TORCH_CHECK((x).scalar_type() == at::kInt, #x " must be int32")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_INT32(x)

namespace {

constexpr int kThreads = 256;

__global__ void count_edges_kernel(
    int const* __restrict__ block_indices,
    int* __restrict__ counts,
    int B,
    int Hp,
    int S,
    int Kb,
    int NB)
{
    int64_t E = (int64_t)B * Hp * S * Kb;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (int64_t e = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; e < E; e += stride) {
        int slot = (int)(e % Kb);
        int64_t tmp = e / Kb;
        int q = (int)(tmp % S);
        tmp /= S;
        int p = (int)(tmp % Hp);
        int b = (int)(tmp / Hp);

        int key_block = block_indices[((int64_t)(b * Hp + p) * S + q) * Kb + slot];
        // Top-k selection may fill early queries with future blocks scored at
        // -inf.  They have no causal contribution, so do not schedule them.
        if ((unsigned)key_block >= (unsigned)NB || key_block > q / 128) {
            continue;
        }
        int bucket = (b * Hp + p) * NB + key_block;
        atomicAdd(counts + bucket, 1);
    }
}

__global__ void scan_fill_meta_kernel(
    int const* __restrict__ counts,
    int* __restrict__ bucket_offsets,
    int* __restrict__ task_meta,
    int* __restrict__ num_tasks,
    int B,
    int Hp,
    int NB,
    int query_chunk,
    int padded_tasks)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    int task = 0;
    int buckets = B * Hp * NB;
    for (int bucket = 0; bucket < buckets; ++bucket) {
        bucket_offsets[bucket] = task;
        int count = counts[bucket];
        int chunks = (count + query_chunk - 1) / query_chunk;
        int key_block = bucket % NB;
        int tmp = bucket / NB;
        int proxy_head = tmp % Hp;
        int batch = tmp / Hp;

        for (int c = 0; c < chunks && task < padded_tasks; ++c) {
            int valid = count - c * query_chunk;
            valid = valid > query_chunk ? query_chunk : valid;
            int64_t row = (int64_t)task * 4;
            task_meta[row + 0] = batch;
            task_meta[row + 1] = proxy_head;
            task_meta[row + 2] = key_block;
            task_meta[row + 3] = valid;
            ++task;
        }
    }
    num_tasks[0] = task;
}

__global__ void scatter_edges_kernel(
    int const* __restrict__ block_indices,
    int const* __restrict__ bucket_offsets,
    int* __restrict__ write_counts,
    int* __restrict__ task_qids,
    int B,
    int Hp,
    int S,
    int Kb,
    int NB,
    int query_chunk,
    int padded_tasks)
{
    int64_t E = (int64_t)B * Hp * S * Kb;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (int64_t e = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; e < E; e += stride) {
        int slot = (int)(e % Kb);
        int64_t tmp = e / Kb;
        int q = (int)(tmp % S);
        tmp /= S;
        int p = (int)(tmp % Hp);
        int b = (int)(tmp / Hp);

        int key_block = block_indices[((int64_t)(b * Hp + p) * S + q) * Kb + slot];
        if ((unsigned)key_block >= (unsigned)NB || key_block > q / 128) {
            continue;
        }

        int bucket = (b * Hp + p) * NB + key_block;
        int local = atomicAdd(write_counts + bucket, 1);
        int task = bucket_offsets[bucket] + local / query_chunk;
        if (task >= padded_tasks) {
            continue;
        }
        int lane = local - (local / query_chunk) * query_chunk;
        task_qids[(int64_t)task * query_chunk + lane] = q;
    }
}

__global__ void count_remote_edges_kernel(
    int const* __restrict__ block_indices,
    int* __restrict__ counts,
    int B,
    int Hp,
    int S,
    int Kb,
    int NB)
{
    int64_t E = (int64_t)B * Hp * S * Kb;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (int64_t e = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; e < E; e += stride) {
        int64_t tmp = e / Kb;
        int q = (int)(tmp % S);
        tmp /= S;
        int p = (int)(tmp % Hp);
        int b = (int)(tmp / Hp);
        int key_block = block_indices[e];
        if ((unsigned)key_block >= (unsigned)NB || key_block >= q / 128) {
            continue;
        }
        int bucket = (b * Hp + p) * NB + key_block;
        atomicAdd(counts + bucket, 1);
    }
}

__global__ void scan_remote_meta_kernel(
    int const* __restrict__ counts,
    int* __restrict__ bucket_offsets,
    int* __restrict__ task_meta,
    int* __restrict__ task_offsets,
    int* __restrict__ sizes,
    int B,
    int Hp,
    int NB,
    int query_chunk,
    int padded_tasks)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    int edge = 0;
    int task = 0;
    int buckets = B * Hp * NB;
    for (int bucket = 0; bucket < buckets; ++bucket) {
        bucket_offsets[bucket] = edge;
        int count = counts[bucket];
        int key_block = bucket % NB;
        int tmp = bucket / NB;
        int proxy_head = tmp % Hp;
        int batch = tmp / Hp;
        for (int start = 0; start < count && task < padded_tasks; start += query_chunk) {
            int valid = min(query_chunk, count - start);
            int64_t row = (int64_t)task * 5;
            task_meta[row + 0] = batch;
            task_meta[row + 1] = proxy_head;
            task_meta[row + 2] = key_block;
            task_meta[row + 3] = valid;
            task_meta[row + 4] = edge + start;
            task_offsets[task] = edge + start;
            ++task;
        }
        edge += count;
    }
    bucket_offsets[buckets] = edge;
    task_offsets[task] = edge;
    sizes[0] = task;
    sizes[1] = edge;
}

__global__ void scatter_remote_edges_kernel(
    int const* __restrict__ block_indices,
    int const* __restrict__ bucket_offsets,
    int* __restrict__ write_counts,
    int* __restrict__ packed_qids,
    int* __restrict__ destinations,
    int* __restrict__ edge_positions,
    int B,
    int Hp,
    int S,
    int Kb,
    int NB)
{
    int64_t E = (int64_t)B * Hp * S * Kb;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (int64_t e = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; e < E; e += stride) {
        int slot = (int)(e % Kb);
        int64_t tmp = e / Kb;
        int q = (int)(tmp % S);
        tmp /= S;
        int p = (int)(tmp % Hp);
        int b = (int)(tmp / Hp);
        int key_block = block_indices[e];
        if ((unsigned)key_block >= (unsigned)NB || key_block >= q / 128) {
            continue;
        }

        int bucket = (b * Hp + p) * NB + key_block;
        int pos = bucket_offsets[bucket] + atomicAdd(write_counts + bucket, 1);
        int destination = (b * Hp + p) * S + q;
        packed_qids[pos] = q;
        destinations[pos] = destination;
        edge_positions[(int64_t)destination * Kb + slot] = pos;
    }
}

__global__ void count_segment_edges_kernel(
    int const* __restrict__ selected_segments,
    int const* __restrict__ token_segments,
    int const* __restrict__ segment_batches,
    int* __restrict__ counts,
    int B,
    int Hp,
    int S,
    int Kb,
    int NS,
    bool remote_only)
{
    int64_t E = (int64_t)B * Hp * S * Kb;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (int64_t e = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; e < E; e += stride) {
        int64_t tmp = e / Kb;
        int q = (int)(tmp % S);
        tmp /= S;
        int p = (int)(tmp % Hp);
        int b = (int)(tmp / Hp);
        int key_segment = selected_segments[e];
        int query_segment = token_segments[(int64_t)b * S + q];
        if ((unsigned)key_segment >= (unsigned)NS
            || segment_batches[key_segment] != b
            || key_segment > query_segment
            || (remote_only && key_segment == query_segment)) {
            continue;
        }
        atomicAdd(counts + p * NS + key_segment, 1);
    }
}

__global__ void scan_segment_meta_kernel(
    int const* __restrict__ counts,
    int const* __restrict__ segment_batches,
    int const* __restrict__ full_segments,
    int* __restrict__ bucket_offsets,
    int* __restrict__ task_meta,
    int* __restrict__ num_tasks,
    int Hp,
    int NS,
    int query_chunk,
    int padded_tasks)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    int task = 0;
    int buckets = Hp * NS;
    for (int full_pass = 1; full_pass >= 0; --full_pass) {
        for (int bucket = 0; bucket < buckets; ++bucket) {
            int key_segment = bucket % NS;
            if (full_segments[key_segment] != full_pass) {
                continue;
            }
            bucket_offsets[bucket] = task;
            int count = counts[bucket];
            int proxy_head = bucket / NS;
            int chunks = (count + query_chunk - 1) / query_chunk;
            for (int c = 0; c < chunks && task < padded_tasks; ++c) {
                int valid = min(query_chunk, count - c * query_chunk);
                int64_t row = (int64_t)task * 4;
                task_meta[row + 0] = segment_batches[key_segment];
                task_meta[row + 1] = proxy_head;
                task_meta[row + 2] = key_segment;
                task_meta[row + 3] = valid;
                ++task;
            }
        }
    }
    num_tasks[0] = task;
}

__global__ void scatter_segment_edges_kernel(
    int const* __restrict__ selected_segments,
    int const* __restrict__ token_segments,
    int const* __restrict__ segment_batches,
    int const* __restrict__ bucket_offsets,
    int* __restrict__ write_counts,
    int* __restrict__ task_qids,
    int B,
    int Hp,
    int S,
    int Kb,
    int NS,
    int query_chunk,
    int padded_tasks)
{
    int64_t E = (int64_t)B * Hp * S * Kb;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (int64_t e = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; e < E; e += stride) {
        int64_t tmp = e / Kb;
        int q = (int)(tmp % S);
        tmp /= S;
        int p = (int)(tmp % Hp);
        int b = (int)(tmp / Hp);
        int key_segment = selected_segments[e];
        int query_segment = token_segments[(int64_t)b * S + q];
        if ((unsigned)key_segment >= (unsigned)NS
            || segment_batches[key_segment] != b
            || key_segment > query_segment) {
            continue;
        }
        int bucket = p * NS + key_segment;
        int local = atomicAdd(write_counts + bucket, 1);
        int task = bucket_offsets[bucket] + local / query_chunk;
        if (task < padded_tasks) {
            int lane = local % query_chunk;
            task_qids[(int64_t)task * query_chunk + lane] = q;
        }
    }
}

__global__ void scan_remote_segment_meta_kernel(
    int const* __restrict__ counts,
    int const* __restrict__ segment_batches,
    int const* __restrict__ full_segments,
    int* __restrict__ bucket_offsets,
    int* __restrict__ task_meta,
    int* __restrict__ task_offsets,
    int* __restrict__ sizes,
    int Hp,
    int NS,
    int query_chunk,
    int padded_tasks)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    int edge = 0;
    int task = 0;
    int buckets = Hp * NS;
    for (int full_pass = 1; full_pass >= 0; --full_pass) {
        for (int bucket = 0; bucket < buckets; ++bucket) {
            int key_segment = bucket % NS;
            if (full_segments[key_segment] != full_pass) {
                continue;
            }
            bucket_offsets[bucket] = edge;
            int count = counts[bucket];
            int proxy_head = bucket / NS;
            for (int start = 0; start < count && task < padded_tasks; start += query_chunk) {
                int valid = min(query_chunk, count - start);
                int64_t row = (int64_t)task * 5;
                task_meta[row + 0] = segment_batches[key_segment];
                task_meta[row + 1] = proxy_head;
                task_meta[row + 2] = key_segment;
                task_meta[row + 3] = valid;
                task_meta[row + 4] = edge + start;
                task_offsets[task] = edge + start;
                ++task;
            }
            edge += count;
        }
    }
    bucket_offsets[buckets] = edge;
    task_offsets[task] = edge;
    sizes[0] = task;
    sizes[1] = edge;
}

__global__ void scatter_remote_segment_edges_kernel(
    int const* __restrict__ selected_segments,
    int const* __restrict__ token_segments,
    int const* __restrict__ segment_batches,
    int const* __restrict__ bucket_offsets,
    int* __restrict__ write_counts,
    int* __restrict__ packed_qids,
    int* __restrict__ destinations,
    int* __restrict__ edge_positions,
    int B,
    int Hp,
    int S,
    int Kb,
    int NS)
{
    int64_t E = (int64_t)B * Hp * S * Kb;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (int64_t e = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; e < E; e += stride) {
        int slot = (int)(e % Kb);
        int64_t tmp = e / Kb;
        int q = (int)(tmp % S);
        tmp /= S;
        int p = (int)(tmp % Hp);
        int b = (int)(tmp / Hp);
        int key_segment = selected_segments[e];
        int query_segment = token_segments[(int64_t)b * S + q];
        if ((unsigned)key_segment >= (unsigned)NS
            || segment_batches[key_segment] != b
            || key_segment >= query_segment) {
            continue;
        }
        int bucket = p * NS + key_segment;
        int pos = bucket_offsets[bucket] + atomicAdd(write_counts + bucket, 1);
        int destination = (b * Hp + p) * S + q;
        packed_qids[pos] = q;
        destinations[pos] = destination;
        edge_positions[(int64_t)destination * Kb + slot] = pos;
    }
}

template <typename scalar_t>
__global__ void merge_attention_chunk_kernel(
    float* __restrict__ output_accum,
    float* __restrict__ lse_accum,
    scalar_t const* __restrict__ remote_output,
    float const* __restrict__ remote_lse,
    int const* __restrict__ destinations,
    int const* __restrict__ edge_positions,
    int edge_start,
    int edge_count,
    int num_destinations,
    int heads_per_proxy,
    int head_dim,
    int top_k_blocks)
{
    int local_edge = (int)blockIdx.x;
    if (local_edge >= edge_count) {
        return;
    }
    int global_edge = edge_start + local_edge;
    int destination = destinations[global_edge];
    if ((unsigned)destination >= (unsigned)num_destinations) {
        return;
    }

    // Only the lowest packed edge for this destination in the current chunk
    // performs the merge.  It consumes every other slot for the same query,
    // eliminating inter-CTA atomics and spin locks.
    for (int slot = 0; slot < top_k_blocks; ++slot) {
        int pos = edge_positions[(int64_t)destination * top_k_blocks + slot];
        if (pos >= edge_start && pos < edge_start + edge_count && pos < global_edge) {
            return;
        }
    }

    __shared__ float old_weight[32];
    __shared__ float edge_weight[32];
    __shared__ float merged_lse[32];
    for (int slot = 0; slot < top_k_blocks; ++slot) {
        int pos = edge_positions[(int64_t)destination * top_k_blocks + slot];
        if (pos < edge_start || pos >= edge_start + edge_count) {
            continue;
        }
        int remote_edge = pos - edge_start;
        if (threadIdx.x < heads_per_proxy) {
            int h = (int)threadIdx.x;
            float old_lse = lse_accum[(int64_t)destination * heads_per_proxy + h];
            float new_lse = remote_lse[(int64_t)remote_edge * heads_per_proxy + h];
            float max_lse = fmaxf(old_lse, new_lse);
            float old_exp = expf(old_lse - max_lse);
            float new_exp = expf(new_lse - max_lse);
            float denom = old_exp + new_exp;
            old_weight[h] = old_exp / denom;
            edge_weight[h] = new_exp / denom;
            merged_lse[h] = logf(denom) + max_lse;
        }
        __syncthreads();

        int values = heads_per_proxy * head_dim;
        for (int linear = (int)threadIdx.x; linear < values; linear += (int)blockDim.x) {
            int h = linear / head_dim;
            int64_t out_idx = (int64_t)destination * values + linear;
            int64_t remote_idx = (int64_t)remote_edge * values + linear;
            output_accum[out_idx] =
                output_accum[out_idx] * old_weight[h]
                + static_cast<float>(remote_output[remote_idx]) * edge_weight[h];
        }
        __syncthreads();
        if (threadIdx.x < heads_per_proxy) {
            int h = (int)threadIdx.x;
            lse_accum[(int64_t)destination * heads_per_proxy + h] = merged_lse[h];
        }
        __syncthreads();
    }
}

__global__ void merge_lse_chunk_kernel(
    float* __restrict__ lse_accum,
    float const* __restrict__ remote_lse,
    int const* __restrict__ destinations,
    int const* __restrict__ edge_positions,
    int edge_start,
    int edge_count,
    int num_destinations,
    int heads_per_proxy,
    int top_k_blocks)
{
    if (threadIdx.x != 0) {
        return;
    }
    int local_edge = (int)blockIdx.x;
    if (local_edge >= edge_count) {
        return;
    }
    int global_edge = edge_start + local_edge;
    int destination = destinations[global_edge];
    if ((unsigned)destination >= (unsigned)num_destinations) {
        return;
    }
    for (int slot = 0; slot < top_k_blocks; ++slot) {
        int pos = edge_positions[(int64_t)destination * top_k_blocks + slot];
        if (pos >= edge_start && pos < edge_start + edge_count && pos < global_edge) {
            return;
        }
    }
    for (int slot = 0; slot < top_k_blocks; ++slot) {
        int pos = edge_positions[(int64_t)destination * top_k_blocks + slot];
        if (pos < edge_start || pos >= edge_start + edge_count) {
            continue;
        }
        int remote_edge = pos - edge_start;
        for (int h = 0; h < heads_per_proxy; ++h) {
            int64_t idx = (int64_t)destination * heads_per_proxy + h;
            float old_lse = lse_accum[idx];
            float new_lse = remote_lse[(int64_t)remote_edge * heads_per_proxy + h];
            float max_lse = fmaxf(old_lse, new_lse);
            lse_accum[idx] = logf(expf(old_lse - max_lse) + expf(new_lse - max_lse)) + max_lse;
        }
    }
}

} // namespace

void run_build_reverse_index(
    torch::Tensor block_indices,
    torch::Tensor counts,
    torch::Tensor write_counts,
    torch::Tensor bucket_offsets,
    torch::Tensor task_meta,
    torch::Tensor task_qids,
    torch::Tensor num_tasks,
    int64_t block_size,
    int64_t query_chunk)
{
    CHECK_INPUT(block_indices);
    CHECK_INPUT(counts);
    CHECK_INPUT(write_counts);
    CHECK_INPUT(bucket_offsets);
    CHECK_INPUT(task_meta);
    CHECK_INPUT(task_qids);
    CHECK_INPUT(num_tasks);

    TORCH_CHECK(block_indices.dim() == 4, "block_indices must have shape [B, Hp, S, Kb]");
    TORCH_CHECK(task_meta.dim() == 2 && task_meta.size(1) == 4, "task_meta must have shape [T, 4]");
    TORCH_CHECK(task_qids.dim() == 2 && task_qids.size(1) == query_chunk, "task_qids must have shape [T, query_chunk]");
    TORCH_CHECK(num_tasks.numel() == 1, "num_tasks must be a scalar tensor");
    TORCH_CHECK(block_size == 128, "backward metadata currently requires block_size=128");
    TORCH_CHECK(query_chunk > 0, "query_chunk must be positive");

    int B = (int)block_indices.size(0);
    int Hp = (int)block_indices.size(1);
    int S = (int)block_indices.size(2);
    int Kb = (int)block_indices.size(3);
    int NB = S / (int)block_size;
    int buckets = B * Hp * NB;
    int padded_tasks = (int)task_meta.size(0);

    TORCH_CHECK(S % block_size == 0, "S must be divisible by block_size");
    TORCH_CHECK(counts.numel() == buckets, "counts has wrong size");
    TORCH_CHECK(write_counts.numel() == buckets, "write_counts has wrong size");
    TORCH_CHECK(bucket_offsets.numel() == buckets, "bucket_offsets has wrong size");
    TORCH_CHECK(task_qids.size(0) == padded_tasks, "task_meta/task_qids row mismatch");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    C10_CUDA_CHECK(cudaMemsetAsync(counts.data_ptr<int>(), 0, counts.numel() * sizeof(int), stream));
    C10_CUDA_CHECK(cudaMemsetAsync(write_counts.data_ptr<int>(), 0, write_counts.numel() * sizeof(int), stream));
    C10_CUDA_CHECK(cudaMemsetAsync(task_meta.data_ptr<int>(), 0, task_meta.numel() * sizeof(int), stream));
    C10_CUDA_CHECK(cudaMemsetAsync(task_qids.data_ptr<int>(), 0xff, task_qids.numel() * sizeof(int), stream));
    C10_CUDA_CHECK(cudaMemsetAsync(num_tasks.data_ptr<int>(), 0, sizeof(int), stream));

    int64_t edges = (int64_t)B * Hp * S * Kb;
    int blocks = (int)std::min<int64_t>((edges + kThreads - 1) / kThreads, 65535);
    blocks = std::max(blocks, 1);

    count_edges_kernel<<<blocks, kThreads, 0, stream>>>(
        block_indices.data_ptr<int>(),
        counts.data_ptr<int>(),
        B,
        Hp,
        S,
        Kb,
        NB);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    scan_fill_meta_kernel<<<1, 1, 0, stream>>>(
        counts.data_ptr<int>(),
        bucket_offsets.data_ptr<int>(),
        task_meta.data_ptr<int>(),
        num_tasks.data_ptr<int>(),
        B,
        Hp,
        NB,
        (int)query_chunk,
        padded_tasks);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    scatter_edges_kernel<<<blocks, kThreads, 0, stream>>>(
        block_indices.data_ptr<int>(),
        bucket_offsets.data_ptr<int>(),
        write_counts.data_ptr<int>(),
        task_qids.data_ptr<int>(),
        B,
        Hp,
        S,
        Kb,
        NB,
        (int)query_chunk,
        padded_tasks);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void run_build_reverse_index_segments(
    torch::Tensor selected_segments,
    torch::Tensor counts,
    torch::Tensor write_counts,
    torch::Tensor bucket_offsets,
    torch::Tensor task_meta,
    torch::Tensor task_qids,
    torch::Tensor num_tasks,
    torch::Tensor token_segments,
    torch::Tensor segment_batches,
    torch::Tensor full_segments,
    int64_t num_segments,
    int64_t query_chunk)
{
    CHECK_INPUT(selected_segments);
    CHECK_INPUT(counts);
    CHECK_INPUT(write_counts);
    CHECK_INPUT(bucket_offsets);
    CHECK_INPUT(task_meta);
    CHECK_INPUT(task_qids);
    CHECK_INPUT(num_tasks);
    CHECK_INPUT(token_segments);
    CHECK_INPUT(segment_batches);
    CHECK_INPUT(full_segments);

    TORCH_CHECK(selected_segments.dim() == 4, "selected_segments must have shape [B, Hp, S, Kb]");
    TORCH_CHECK(token_segments.dim() == 2, "token_segments must have shape [B, S]");
    TORCH_CHECK(task_meta.dim() == 2 && task_meta.size(1) == 4, "task_meta must have shape [T, 4]");
    TORCH_CHECK(query_chunk > 0 && num_segments > 0, "query_chunk and num_segments must be positive");

    int B = (int)selected_segments.size(0);
    int Hp = (int)selected_segments.size(1);
    int S = (int)selected_segments.size(2);
    int Kb = (int)selected_segments.size(3);
    int NS = (int)num_segments;
    int buckets = Hp * NS;
    int padded_tasks = (int)task_meta.size(0);
    TORCH_CHECK(token_segments.size(0) == B && token_segments.size(1) == S, "token_segments shape mismatch");
    TORCH_CHECK(segment_batches.numel() == NS, "segment_batches has wrong size");
    TORCH_CHECK(full_segments.numel() == NS, "full_segments has wrong size");
    TORCH_CHECK(counts.numel() == buckets && write_counts.numel() == buckets, "counts have wrong size");
    TORCH_CHECK(bucket_offsets.numel() == buckets, "bucket_offsets has wrong size");
    TORCH_CHECK(task_qids.size(0) == padded_tasks && task_qids.size(1) == query_chunk, "task_qids shape mismatch");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    C10_CUDA_CHECK(cudaMemsetAsync(counts.data_ptr<int>(), 0, counts.numel() * sizeof(int), stream));
    C10_CUDA_CHECK(cudaMemsetAsync(write_counts.data_ptr<int>(), 0, write_counts.numel() * sizeof(int), stream));
    C10_CUDA_CHECK(cudaMemsetAsync(task_meta.data_ptr<int>(), 0, task_meta.numel() * sizeof(int), stream));
    C10_CUDA_CHECK(cudaMemsetAsync(task_qids.data_ptr<int>(), 0xff, task_qids.numel() * sizeof(int), stream));
    C10_CUDA_CHECK(cudaMemsetAsync(num_tasks.data_ptr<int>(), 0, sizeof(int), stream));

    int64_t edges = (int64_t)B * Hp * S * Kb;
    int blocks = (int)std::min<int64_t>((edges + kThreads - 1) / kThreads, 65535);
    blocks = std::max(blocks, 1);
    count_segment_edges_kernel<<<blocks, kThreads, 0, stream>>>(
        selected_segments.data_ptr<int>(), token_segments.data_ptr<int>(),
        segment_batches.data_ptr<int>(), counts.data_ptr<int>(), B, Hp, S, Kb, NS, false);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    scan_segment_meta_kernel<<<1, 1, 0, stream>>>(
        counts.data_ptr<int>(), segment_batches.data_ptr<int>(), full_segments.data_ptr<int>(),
        bucket_offsets.data_ptr<int>(),
        task_meta.data_ptr<int>(), num_tasks.data_ptr<int>(), Hp, NS, (int)query_chunk, padded_tasks);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    scatter_segment_edges_kernel<<<blocks, kThreads, 0, stream>>>(
        selected_segments.data_ptr<int>(), token_segments.data_ptr<int>(),
        segment_batches.data_ptr<int>(), bucket_offsets.data_ptr<int>(), write_counts.data_ptr<int>(),
        task_qids.data_ptr<int>(), B, Hp, S, Kb, NS, (int)query_chunk, padded_tasks);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void run_build_remote_metadata(
    torch::Tensor block_indices,
    torch::Tensor counts,
    torch::Tensor write_counts,
    torch::Tensor bucket_offsets,
    torch::Tensor task_meta,
    torch::Tensor task_offsets,
    torch::Tensor packed_qids,
    torch::Tensor destinations,
    torch::Tensor edge_positions,
    torch::Tensor sizes,
    int64_t block_size,
    int64_t query_chunk)
{
    CHECK_INPUT(block_indices);
    CHECK_INPUT(counts);
    CHECK_INPUT(write_counts);
    CHECK_INPUT(bucket_offsets);
    CHECK_INPUT(task_meta);
    CHECK_INPUT(task_offsets);
    CHECK_INPUT(packed_qids);
    CHECK_INPUT(destinations);
    CHECK_INPUT(edge_positions);
    CHECK_INPUT(sizes);

    TORCH_CHECK(block_size == 128, "remote metadata currently requires block_size=128");
    TORCH_CHECK(query_chunk > 0, "query_chunk must be positive");
    TORCH_CHECK(block_indices.dim() == 4, "block_indices must have shape [B, Hp, S, Kb]");
    TORCH_CHECK(task_meta.dim() == 2 && task_meta.size(1) == 5, "task_meta must have shape [T, 5]");
    TORCH_CHECK(sizes.numel() == 2, "sizes must contain [num_tasks, num_edges]");

    int B = (int)block_indices.size(0);
    int Hp = (int)block_indices.size(1);
    int S = (int)block_indices.size(2);
    int Kb = (int)block_indices.size(3);
    int NB = S / (int)block_size;
    int buckets = B * Hp * NB;
    int padded_tasks = (int)task_meta.size(0);
    int64_t max_edges = (int64_t)B * Hp * S * Kb;

    TORCH_CHECK(S % block_size == 0, "S must be divisible by block_size");
    TORCH_CHECK(counts.numel() == buckets && write_counts.numel() == buckets, "remote counts have wrong size");
    TORCH_CHECK(bucket_offsets.numel() == buckets + 1, "bucket_offsets has wrong size");
    TORCH_CHECK(task_offsets.numel() == padded_tasks + 1, "task_offsets has wrong size");
    TORCH_CHECK(packed_qids.numel() == max_edges && destinations.numel() == max_edges, "packed edge buffers have wrong size");
    TORCH_CHECK(edge_positions.numel() == max_edges, "edge_positions has wrong size");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    C10_CUDA_CHECK(cudaMemsetAsync(counts.data_ptr<int>(), 0, counts.numel() * sizeof(int), stream));
    C10_CUDA_CHECK(cudaMemsetAsync(write_counts.data_ptr<int>(), 0, write_counts.numel() * sizeof(int), stream));
    C10_CUDA_CHECK(cudaMemsetAsync(task_meta.data_ptr<int>(), 0, task_meta.numel() * sizeof(int), stream));
    C10_CUDA_CHECK(cudaMemsetAsync(task_offsets.data_ptr<int>(), 0, task_offsets.numel() * sizeof(int), stream));
    C10_CUDA_CHECK(cudaMemsetAsync(packed_qids.data_ptr<int>(), 0xff, packed_qids.numel() * sizeof(int), stream));
    C10_CUDA_CHECK(cudaMemsetAsync(destinations.data_ptr<int>(), 0xff, destinations.numel() * sizeof(int), stream));
    C10_CUDA_CHECK(cudaMemsetAsync(edge_positions.data_ptr<int>(), 0xff, edge_positions.numel() * sizeof(int), stream));
    C10_CUDA_CHECK(cudaMemsetAsync(sizes.data_ptr<int>(), 0, sizes.numel() * sizeof(int), stream));

    int blocks = (int)std::min<int64_t>((max_edges + kThreads - 1) / kThreads, 65535);
    blocks = std::max(blocks, 1);
    count_remote_edges_kernel<<<blocks, kThreads, 0, stream>>>(
        block_indices.data_ptr<int>(), counts.data_ptr<int>(), B, Hp, S, Kb, NB);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    scan_remote_meta_kernel<<<1, 1, 0, stream>>>(
        counts.data_ptr<int>(), bucket_offsets.data_ptr<int>(), task_meta.data_ptr<int>(),
        task_offsets.data_ptr<int>(), sizes.data_ptr<int>(), B, Hp, NB,
        (int)query_chunk, padded_tasks);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    scatter_remote_edges_kernel<<<blocks, kThreads, 0, stream>>>(
        block_indices.data_ptr<int>(), bucket_offsets.data_ptr<int>(), write_counts.data_ptr<int>(),
        packed_qids.data_ptr<int>(), destinations.data_ptr<int>(), edge_positions.data_ptr<int>(),
        B, Hp, S, Kb, NB);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void run_build_remote_metadata_segments(
    torch::Tensor selected_segments,
    torch::Tensor counts,
    torch::Tensor write_counts,
    torch::Tensor bucket_offsets,
    torch::Tensor task_meta,
    torch::Tensor task_offsets,
    torch::Tensor packed_qids,
    torch::Tensor destinations,
    torch::Tensor edge_positions,
    torch::Tensor sizes,
    torch::Tensor token_segments,
    torch::Tensor segment_batches,
    torch::Tensor full_segments,
    int64_t num_segments,
    int64_t query_chunk)
{
    CHECK_INPUT(selected_segments);
    CHECK_INPUT(counts);
    CHECK_INPUT(write_counts);
    CHECK_INPUT(bucket_offsets);
    CHECK_INPUT(task_meta);
    CHECK_INPUT(task_offsets);
    CHECK_INPUT(packed_qids);
    CHECK_INPUT(destinations);
    CHECK_INPUT(edge_positions);
    CHECK_INPUT(sizes);
    CHECK_INPUT(token_segments);
    CHECK_INPUT(segment_batches);
    CHECK_INPUT(full_segments);

    TORCH_CHECK(selected_segments.dim() == 4, "selected_segments must have shape [B, Hp, S, Kb]");
    TORCH_CHECK(task_meta.dim() == 2 && task_meta.size(1) == 5, "task_meta must have shape [T, 5]");
    TORCH_CHECK(query_chunk > 0 && num_segments > 0, "query_chunk and num_segments must be positive");

    int B = (int)selected_segments.size(0);
    int Hp = (int)selected_segments.size(1);
    int S = (int)selected_segments.size(2);
    int Kb = (int)selected_segments.size(3);
    int NS = (int)num_segments;
    int buckets = Hp * NS;
    int padded_tasks = (int)task_meta.size(0);
    int64_t max_edges = (int64_t)B * Hp * S * Kb;
    TORCH_CHECK(token_segments.dim() == 2 && token_segments.size(0) == B && token_segments.size(1) == S, "token_segments shape mismatch");
    TORCH_CHECK(segment_batches.numel() == NS, "segment_batches has wrong size");
    TORCH_CHECK(full_segments.numel() == NS, "full_segments has wrong size");
    TORCH_CHECK(counts.numel() == buckets && write_counts.numel() == buckets, "counts have wrong size");
    TORCH_CHECK(bucket_offsets.numel() == buckets + 1, "bucket_offsets has wrong size");
    TORCH_CHECK(task_offsets.numel() == padded_tasks + 1, "task_offsets has wrong size");
    TORCH_CHECK(packed_qids.numel() == max_edges && destinations.numel() == max_edges, "packed edge buffers have wrong size");
    TORCH_CHECK(edge_positions.numel() == max_edges, "edge_positions has wrong size");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    C10_CUDA_CHECK(cudaMemsetAsync(counts.data_ptr<int>(), 0, counts.numel() * sizeof(int), stream));
    C10_CUDA_CHECK(cudaMemsetAsync(write_counts.data_ptr<int>(), 0, write_counts.numel() * sizeof(int), stream));
    C10_CUDA_CHECK(cudaMemsetAsync(task_meta.data_ptr<int>(), 0, task_meta.numel() * sizeof(int), stream));
    C10_CUDA_CHECK(cudaMemsetAsync(task_offsets.data_ptr<int>(), 0, task_offsets.numel() * sizeof(int), stream));
    C10_CUDA_CHECK(cudaMemsetAsync(packed_qids.data_ptr<int>(), 0xff, packed_qids.numel() * sizeof(int), stream));
    C10_CUDA_CHECK(cudaMemsetAsync(destinations.data_ptr<int>(), 0xff, destinations.numel() * sizeof(int), stream));
    C10_CUDA_CHECK(cudaMemsetAsync(edge_positions.data_ptr<int>(), 0xff, edge_positions.numel() * sizeof(int), stream));
    C10_CUDA_CHECK(cudaMemsetAsync(sizes.data_ptr<int>(), 0, sizes.numel() * sizeof(int), stream));

    int blocks = (int)std::min<int64_t>((max_edges + kThreads - 1) / kThreads, 65535);
    blocks = std::max(blocks, 1);
    count_segment_edges_kernel<<<blocks, kThreads, 0, stream>>>(
        selected_segments.data_ptr<int>(), token_segments.data_ptr<int>(),
        segment_batches.data_ptr<int>(), counts.data_ptr<int>(), B, Hp, S, Kb, NS, true);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    scan_remote_segment_meta_kernel<<<1, 1, 0, stream>>>(
        counts.data_ptr<int>(), segment_batches.data_ptr<int>(), full_segments.data_ptr<int>(),
        bucket_offsets.data_ptr<int>(),
        task_meta.data_ptr<int>(), task_offsets.data_ptr<int>(), sizes.data_ptr<int>(),
        Hp, NS, (int)query_chunk, padded_tasks);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    scatter_remote_segment_edges_kernel<<<blocks, kThreads, 0, stream>>>(
        selected_segments.data_ptr<int>(), token_segments.data_ptr<int>(),
        segment_batches.data_ptr<int>(), bucket_offsets.data_ptr<int>(), write_counts.data_ptr<int>(),
        packed_qids.data_ptr<int>(), destinations.data_ptr<int>(), edge_positions.data_ptr<int>(),
        B, Hp, S, Kb, NS);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void merge_attention_chunk_cuda(
    torch::Tensor output_accum,
    torch::Tensor lse_accum,
    torch::Tensor remote_output,
    torch::Tensor remote_lse,
    torch::Tensor destinations,
    torch::Tensor edge_positions,
    int64_t edge_start,
    int64_t top_k_blocks)
{
    CHECK_CUDA(output_accum);
    CHECK_CUDA(lse_accum);
    CHECK_CUDA(remote_output);
    CHECK_CUDA(remote_lse);
    CHECK_INPUT(destinations);
    CHECK_INPUT(edge_positions);
    CHECK_CONTIGUOUS(output_accum);
    CHECK_CONTIGUOUS(lse_accum);
    CHECK_CONTIGUOUS(remote_output);
    CHECK_CONTIGUOUS(remote_lse);
    TORCH_CHECK(output_accum.scalar_type() == at::kFloat && lse_accum.scalar_type() == at::kFloat, "accumulators must be FP32");
    TORCH_CHECK(remote_lse.scalar_type() == at::kFloat, "remote_lse must be FP32");
    TORCH_CHECK(remote_output.scalar_type() == at::kHalf || remote_output.scalar_type() == at::kBFloat16, "remote_output must be FP16/BF16");
    TORCH_CHECK(output_accum.dim() == 3 && lse_accum.dim() == 2 && remote_output.dim() == 3 && remote_lse.dim() == 2, "invalid merge tensor ranks");
    int edge_count = (int)remote_output.size(0);
    int num_destinations = (int)output_accum.size(0);
    int heads_per_proxy = (int)output_accum.size(1);
    int head_dim = (int)output_accum.size(2);
    TORCH_CHECK(heads_per_proxy <= 32, "merge supports at most 32 heads per proxy");
    TORCH_CHECK(lse_accum.size(0) == num_destinations && lse_accum.size(1) == heads_per_proxy, "lse_accum shape mismatch");
    TORCH_CHECK(remote_output.size(1) == heads_per_proxy && remote_output.size(2) == head_dim, "remote_output shape mismatch");
    TORCH_CHECK(remote_lse.size(0) == edge_count && remote_lse.size(1) == heads_per_proxy, "remote_lse shape mismatch");
    if (edge_count == 0) {
        return;
    }
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (remote_output.scalar_type() == at::kHalf) {
        merge_attention_chunk_kernel<at::Half><<<edge_count, 256, 0, stream>>>(
            output_accum.data_ptr<float>(), lse_accum.data_ptr<float>(),
            remote_output.data_ptr<at::Half>(), remote_lse.data_ptr<float>(),
            destinations.data_ptr<int>(), edge_positions.data_ptr<int>(),
            (int)edge_start, edge_count, num_destinations, heads_per_proxy,
            head_dim, (int)top_k_blocks);
    } else {
        merge_attention_chunk_kernel<at::BFloat16><<<edge_count, 256, 0, stream>>>(
            output_accum.data_ptr<float>(), lse_accum.data_ptr<float>(),
            remote_output.data_ptr<at::BFloat16>(), remote_lse.data_ptr<float>(),
            destinations.data_ptr<int>(), edge_positions.data_ptr<int>(),
            (int)edge_start, edge_count, num_destinations, heads_per_proxy,
            head_dim, (int)top_k_blocks);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void merge_lse_chunk_cuda(
    torch::Tensor lse_accum,
    torch::Tensor remote_lse,
    torch::Tensor destinations,
    torch::Tensor edge_positions,
    int64_t edge_start,
    int64_t top_k_blocks)
{
    CHECK_CUDA(lse_accum);
    CHECK_CUDA(remote_lse);
    CHECK_INPUT(destinations);
    CHECK_INPUT(edge_positions);
    CHECK_CONTIGUOUS(lse_accum);
    CHECK_CONTIGUOUS(remote_lse);
    TORCH_CHECK(lse_accum.scalar_type() == at::kFloat && remote_lse.scalar_type() == at::kFloat, "LSE tensors must be FP32");
    int edge_count = (int)remote_lse.size(0);
    int num_destinations = (int)lse_accum.size(0);
    int heads_per_proxy = (int)lse_accum.size(1);
    TORCH_CHECK(remote_lse.dim() == 2 && remote_lse.size(1) == heads_per_proxy, "remote_lse shape mismatch");
    if (edge_count == 0) {
        return;
    }
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    merge_lse_chunk_kernel<<<edge_count, 32, 0, stream>>>(
        lse_accum.data_ptr<float>(), remote_lse.data_ptr<float>(),
        destinations.data_ptr<int>(), edge_positions.data_ptr<int>(),
        (int)edge_start, edge_count, num_destinations, heads_per_proxy,
        (int)top_k_blocks);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run_build_reverse_index", &run_build_reverse_index, "Build MSA reverse index on CUDA");
    m.def("run_build_reverse_index_segments", &run_build_reverse_index_segments, "Build document-segment MSA reverse index on CUDA");
    m.def("run_build_remote_metadata", &run_build_remote_metadata, "Build compact remote-edge metadata on CUDA");
    m.def("run_build_remote_metadata_segments", &run_build_remote_metadata_segments, "Build compact document-segment remote metadata on CUDA");
    m.def("merge_attention_chunk", &merge_attention_chunk_cuda, "Online-merge one varlen attention chunk");
    m.def("merge_lse_chunk", &merge_lse_chunk_cuda, "Online-merge one varlen LSE chunk");
}
