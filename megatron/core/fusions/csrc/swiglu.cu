#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/IntegerDivider.cuh>
#include <ATen/ExpandUtils.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <type_traits>
#include <vector>

namespace {
constexpr int kThreads = 128;
constexpr int kGridRows = 65535;

template <typename T, int V>
struct alignas(sizeof(T) * V) Pack { T data[V]; };

// Reduce across the block to avoid nondeterministic atomic accumulation of weight gradients.
__device__ __forceinline__ float block_sum(float value) {
    __shared__ float sums[32];
    const int lane = threadIdx.x % warpSize;
    const int warp = threadIdx.x / warpSize;
    for (int delta = warpSize / 2; delta; delta /= 2)
        value += __shfl_down_sync(0xffffffff, value, delta);
    if (lane == 0) sums[warp] = value;
    __syncthreads();
    value = threadIdx.x < kThreads / warpSize ? sums[lane] : 0.f;
    if (warp == 0) {
        for (int delta = warpSize / 2; delta; delta /= 2)
            value += __shfl_down_sync(0xffffffff, value, delta);
    }
    return value;
}

// Share vectorized accesses across forward, plain backward, and weighted backward.
// Remove unused branches at compile time; tile by token, or use one warp per row for H=512.
template <typename T, typename W, int V, bool Back, bool Weighted, bool Bias,
          int RowThreads = kThreads, int FixedH = 0, bool DirectTile = false>
__global__ __launch_bounds__(kThreads, 4) void swiglu_kernel(
    const T* __restrict__ y, const T* __restrict__ g,
    const T* __restrict__ bias, const W* __restrict__ weights,
    T* __restrict__ out, W* __restrict__ dw, float* __restrict__ partial,
    int64_t dynamic_h, int64_t tiles, int64_t bias_stride, int64_t rows) {
    constexpr int RowsPerBlock = kThreads / RowThreads;
    const int64_t h = FixedH ? FixedH : dynamic_h;
    const int64_t row = DirectTile
        ? int64_t(blockIdx.y) + int64_t(blockIdx.z) * kGridRows
        : FixedH ? int64_t(blockIdx.x) * RowsPerBlock + threadIdx.x / RowThreads
                 : int64_t(blockIdx.x) / tiles;
    const int64_t tile = DirectTile ? int64_t(blockIdx.x)
                                   : FixedH ? 0 : int64_t(blockIdx.x) % tiles;
    if constexpr (FixedH || DirectTile) {
        if (row >= rows) return;
    }
    const int64_t vectors = h / V;
    const int64_t tile_vectors = Back && Weighted ? 4 * kThreads : kThreads;
    const int64_t begin = FixedH ? 0 : tile * tile_vectors;
    const int64_t end = FixedH || (Back && Weighted && tiles == 1)
        ? vectors : min(vectors, begin + tile_vectors);
    const float weight = Weighted ? float(weights[row]) : 1.f;
    // Use FP64 reduction for FP32 weights on the H=512 warp path; group Packs to shorten FP32 chains.
    using WeightAcc = std::conditional_t<Back && Weighted && RowThreads == 32 &&
                                        std::is_same_v<W, float>, double, float>;
    constexpr bool GroupedWeight = Back && Weighted && RowThreads == 32 &&
                                    std::is_same_v<W, float> && V == 8;
    float weight_bins[GroupedWeight ? V : 1] = {};
    WeightAcc weight_sum = 0;
    #pragma unroll
    for (int64_t v = begin + threadIdx.x % RowThreads; v < end; v += RowThreads) {
        const int64_t col = v * V;
        const int64_t offset = row * 2 * h + col;
        Pack<T, V> a = *reinterpret_cast<const Pack<T, V>*>(y + offset);
        Pack<T, V> b = *reinterpret_cast<const Pack<T, V>*>(y + offset + h);
        Pack<T, V> ba, bb, grad, oa, ob;
        if constexpr (Bias) {
            const int64_t bo = row * bias_stride + col;
            ba = *reinterpret_cast<const Pack<T, V>*>(bias + bo);
            bb = *reinterpret_cast<const Pack<T, V>*>(bias + bo + h);
        }
        if constexpr (Back)
            grad = *reinterpret_cast<const Pack<T, V>*>(g + row * h + col);
        #pragma unroll
        for (int j = 0; j < V; ++j) {
            float x = float(a.data[j]);
            float z = float(b.data[j]);
            if constexpr (Bias) {
                // Match fused compilation: keep bias addition in FP32 without early input-dtype rounding.
                x += float(ba.data[j]);
                z += float(bb.data[j]);
            }
            const float s = 1.f / (1.f + __expf(-x));
            const float silu = x * s;
            if constexpr (Back) {
                const float go = float(grad.data[j]);
                const float scaled = go * weight;
                oa.data[j] = T(scaled * s * (1.f + x * (1.f - s)) * z);
                ob.data[j] = T(scaled * silu);
                if constexpr (Weighted) {
                    // Keep FP32 intermediates in the fused path; avoid BF16/FP16 rounding before reduction.
                    if constexpr (GroupedWeight) weight_bins[j] += (silu * z) * go;
                    else weight_sum += WeightAcc((silu * z) * go);
                }
            } else {
                float activation = silu * z;
                if constexpr (Weighted) activation *= weight;
                oa.data[j] = T(activation);
            }
        }
        if constexpr (Back) {
            *reinterpret_cast<Pack<T, V>*>(out + offset) = oa;
            *reinterpret_cast<Pack<T, V>*>(out + offset + h) = ob;
        } else {
            *reinterpret_cast<Pack<T, V>*>(out + row * h + col) = oa;
        }
    }
    if constexpr (Back && Weighted) {
        if constexpr (GroupedWeight) {
            #pragma unroll
            for (int j = 0; j < V; ++j) weight_sum += WeightAcc(weight_bins[j]);
        }
        WeightAcc total;
        if constexpr (RowThreads == 32) {
            // All warp lanes participate without shared memory or a CTA barrier.
            for (int delta = 16; delta; delta /= 2)
                weight_sum += __shfl_down_sync(0xffffffff, weight_sum, delta);
            total = weight_sum;
        } else {
            total = block_sum(weight_sum);
        }
        if (threadIdx.x % RowThreads == 0) {
            if (tiles == 1) dw[row] = W(total);
            else partial[row * tiles + tile] = total;
        }
    }
}

// Unroll independent wide-row elements to amortize CTA indexing; retain naturally aligned tails.
template <typename T, typename W, bool Weighted, bool Bias, int Items>
__global__ __launch_bounds__(kThreads, 4) void swiglu_forward_scalar_kernel(
    const T* __restrict__ y, const T* __restrict__ bias,
    const W* __restrict__ weights, T* __restrict__ out,
    int64_t h, int64_t bias_stride, int64_t rows) {
    const int64_t row = int64_t(blockIdx.y) + int64_t(blockIdx.z) * kGridRows;
    if (row >= rows) return;
    const int64_t begin = int64_t(blockIdx.x) * (Items * kThreads) + threadIdx.x;
    const float weight = Weighted ? float(weights[row]) : 1.f;
    T a[Items], b[Items], ba[Items], bb[Items];
    #pragma unroll
    for (int i = 0; i < Items; ++i) {
        const int64_t col = begin + i * kThreads;
        if (col < h) {
            const int64_t offset = row * 2 * h + col;
            a[i] = y[offset];
            b[i] = y[offset + h];
            if constexpr (Bias) {
                const int64_t bo = row * bias_stride + col;
                ba[i] = bias[bo];
                bb[i] = bias[bo + h];
            }
        }
    }
    #pragma unroll
    for (int i = 0; i < Items; ++i) {
        const int64_t col = begin + i * kThreads;
        if (col < h) {
            float x = float(a[i]), z = float(b[i]);
            if constexpr (Bias) {
                x += float(ba[i]);
                z += float(bb[i]);
            }
            const float s = 1.f / (1.f + __expf(-x));
            const float silu = x * s;
            float activation = silu * z;
            if constexpr (Weighted) activation *= weight;
            out[row * h + col] = T(activation);
        }
    }
}

// Keep scalar FP16 loads naturally aligned and store pairs, reading both rows for crossing pairs.
template <typename W, bool Weighted, bool Even>
__global__ __launch_bounds__(256, 4) void swiglu_forward_pair_kernel(
    const at::Half* __restrict__ y, const W* __restrict__ weights,
    at::Half* __restrict__ out, at::cuda::detail::IntDivider<unsigned int> h_divider,
    int elements) {
    const int index = (int(blockIdx.x) * 256 + threadIdx.x) * 2;
    if (index >= elements) return;
    // Eligibility limits input elements to INT32_MAX, keeping valid indices within fast-division limits.
    const auto row_col = h_divider.divmod(static_cast<unsigned int>(index));
    const int h = int(h_divider.divisor), row = int(row_col.div), col = int(row_col.mod);
    const int offset = index + row * h;
    // Even output sizes always have a second element; guard the final element for odd sizes.
    const bool next = Even || index + 1 < elements, cross_row = col + 1 == h;
    const int next_offset = offset + 1 + (cross_row ? h : 0);
    const float a0 = float(y[offset]), b0 = float(y[offset + h]);
    const float a1 = next ? float(y[next_offset]) : 0.f;
    const float b1 = next ? float(y[next_offset + h]) : 0.f;
    const float w0 = Weighted ? float(weights[row]) : 1.f;
    const float w1 = Weighted && next ? float(weights[row + int(cross_row)]) : 1.f;
    const float s0 = 1.f / (1.f + __expf(-a0));
    const float s1 = 1.f / (1.f + __expf(-a1));
    Pack<at::Half, 2> result;
    float v0 = (a0 * s0) * b0, v1 = (a1 * s1) * b1;
    if constexpr (Weighted) {
        v0 *= w0;
        v1 *= w1;
    }
    result.data[0] = at::Half(v0);
    result.data[1] = at::Half(v1);
    if (next) *reinterpret_cast<Pack<at::Half, 2>*>(out + index) = result;
    else out[index] = result.data[0];
}

// Process exactly one Pack per thread without a dynamic loop that can iterate at most once.
template <typename T, typename W, int V, bool Weighted, bool Bias>
__global__ __launch_bounds__(kThreads, 4) void swiglu_forward_pack_kernel(
    const T* __restrict__ y, const T* __restrict__ bias,
    const W* __restrict__ weights, T* __restrict__ out,
    int64_t h, int64_t bias_stride, int64_t rows) {
    const int64_t row = int64_t(blockIdx.y) + int64_t(blockIdx.z) * kGridRows;
    const int64_t col = (int64_t(blockIdx.x) * kThreads + threadIdx.x) * V;
    if (row >= rows || col >= h) return;
    const int64_t offset = row * 2 * h + col;
    const Pack<T, V> a = *reinterpret_cast<const Pack<T, V>*>(y + offset);
    const Pack<T, V> b = *reinterpret_cast<const Pack<T, V>*>(y + offset + h);
    Pack<T, V> ba, bb, output;
    if constexpr (Bias) {
        const int64_t bo = row * bias_stride + col;
        ba = *reinterpret_cast<const Pack<T, V>*>(bias + bo);
        bb = *reinterpret_cast<const Pack<T, V>*>(bias + bo + h);
    }
    const float weight = Weighted ? float(weights[row]) : 1.f;
    #pragma unroll
    for (int j = 0; j < V; ++j) {
        float x = float(a.data[j]), z = float(b.data[j]);
        if constexpr (Bias) {
            x += float(ba.data[j]);
            z += float(bb.data[j]);
        }
        const float s = 1.f / (1.f + __expf(-x));
        const float silu = x * s;
        float activation = silu * z;
        if constexpr (Weighted) activation *= weight;
        output.data[j] = T(activation);
    }
    *reinterpret_cast<Pack<T, V>*>(out + row * h + col) = output;
}

// Tile contiguous output Packs across rows; static widths eliminate dynamic division.
// Use 32-bit indices only up to INT32_MAX input elements; preserve the other path for larger inputs.
template <typename T, typename W, int V, bool Weighted, bool Bias, int H>
__global__ __launch_bounds__(kThreads, 4) void swiglu_forward_flat_kernel(
    const T* __restrict__ y, const T* __restrict__ bias,
    const W* __restrict__ weights, T* __restrict__ out,
    int packs, int bias_stride) {
    const int index = int(blockIdx.x) * kThreads + threadIdx.x;
    if (index >= packs) return;
    constexpr int RowPacks = H / V;
    const int row = index / RowPacks;
    const int col = (index % RowPacks) * V;
    const int offset = row * (2 * H) + col;
    const Pack<T, V> a = *reinterpret_cast<const Pack<T, V>*>(y + offset);
    const Pack<T, V> b = *reinterpret_cast<const Pack<T, V>*>(y + offset + H);
    Pack<T, V> ba, bb, output;
    if constexpr (Bias) {
        const int bo = row * bias_stride + col;
        ba = *reinterpret_cast<const Pack<T, V>*>(bias + bo);
        bb = *reinterpret_cast<const Pack<T, V>*>(bias + bo + H);
    }
    const float weight = Weighted ? float(weights[row]) : 1.f;
    #pragma unroll
    for (int j = 0; j < V; ++j) {
        float x = float(a.data[j]);
        float z = float(b.data[j]);
        if constexpr (Bias) {
            x += float(ba.data[j]);
            z += float(bb.data[j]);
        }
        const float s = 1.f / (1.f + __expf(-x));
        const float silu = x * s;
        float activation = silu * z;
        if constexpr (Weighted) activation *= weight;
        output.data[j] = T(activation);
    }
    reinterpret_cast<Pack<T, V>*>(out)[index] = output;
}

template <typename T, typename W, int V, bool Weighted, bool Bias, int H>
void launch_forward_flat(
    const torch::Tensor& y, const torch::Tensor& bias, const torch::Tensor& weights,
    torch::Tensor& out, int64_t bias_stride, cudaStream_t stream) {
    const int packs = int(out.numel() / V);
    swiglu_forward_flat_kernel<T, W, V, Weighted, Bias, H>
        <<<(packs + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
            y.data_ptr<T>(), Bias ? bias.data_ptr<T>() : nullptr,
            Weighted ? weights.data_ptr<W>() : nullptr, out.data_ptr<T>(),
            packs, int(bias_stride));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Launch a second reduction only for split wide rows, using an FP32 [tokens, tiles] workspace.
template <typename W>
__global__ void finish_weight_grad(
    const float* __restrict__ partial, W* __restrict__ dw,
    int64_t rows, int64_t tiles) {
    const int64_t row = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row < rows) {
        float sum = 0.f;
        for (int64_t i = 0; i < tiles; ++i) sum += partial[row * tiles + i];
        dw[row] = W(sum);
    }
}

bool access_aligned(const torch::Tensor& t, int64_t bytes, int64_t row_stride) {
    return !t.defined() ||
        (reinterpret_cast<uintptr_t>(t.data_ptr()) % bytes == 0 && row_stride % bytes == 0);
}

template <typename T, typename W, int V, bool Back, bool Weighted, bool Bias>
void launch(
    const torch::Tensor& y, const torch::Tensor& g, const torch::Tensor& bias,
    const torch::Tensor& weights, torch::Tensor& out, torch::Tensor& dw,
    int64_t rows, int64_t h, int64_t bias_stride, cudaStream_t stream) {
    if constexpr (!Back && V == 8 && std::is_same_v<T, at::BFloat16>) {
        if ((h == 768 || h == 14336) && y.numel() <= std::numeric_limits<int32_t>::max()) {
            if (h == 768) {
                launch_forward_flat<T, W, V, Weighted, Bias, 768>(
                    y, bias, weights, out, bias_stride, stream);
                return;
            }
            if (h == 14336) {
                launch_forward_flat<T, W, V, Weighted, Bias, 14336>(
                    y, bias, weights, out, bias_stride, stream);
                return;
            }
        }
    }
    if constexpr (V == 16 / sizeof(T)) {
        if (h == 512 && rows >= kThreads) {
            constexpr int RowThreads = 32;
            constexpr int RowsPerBlock = kThreads / RowThreads;
            TORCH_CHECK((rows + RowsPerBlock - 1) / RowsPerBlock <=
                        std::numeric_limits<int32_t>::max(), "SwiGLU launch grid is too large");
            swiglu_kernel<T, W, V, Back, Weighted, Bias, RowThreads, 512>
                <<<(rows + RowsPerBlock - 1) / RowsPerBlock, kThreads, 0, stream>>>(
                    y.data_ptr<T>(), Back ? g.data_ptr<T>() : nullptr,
                    Bias ? bias.data_ptr<T>() : nullptr,
                    Weighted ? weights.data_ptr<W>() : nullptr, out.data_ptr<T>(),
                    Back && Weighted ? dw.data_ptr<W>() : nullptr, nullptr,
                    h, 1, bias_stride, rows);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            return;
        }
    }
    const int64_t vectors = h / V;
    // Parallelize elementwise work by tile; keep reductions in one CTA when there are enough tokens.
    const int64_t tiles = Back && Weighted
        ? (rows >= 256 ? 1 : (vectors + 4 * kThreads - 1) / (4 * kThreads))
        : (vectors + kThreads - 1) / kThreads;
    TORCH_CHECK(rows <= std::numeric_limits<int32_t>::max() / tiles,
                "SwiGLU launch grid is too large");
    if constexpr (!Back) {
        if constexpr (V == 1) {
            if constexpr (std::is_same_v<T, at::Half> && !Bias) {
                if (h >= 512 && y.numel() <= std::numeric_limits<int32_t>::max() &&
                    access_aligned(out, 2 * sizeof(T), 0)) {
                    const auto h_divider = at::cuda::detail::IntDivider<unsigned int>(
                        static_cast<unsigned int>(h));
                    if (out.numel() % 2 == 0) {
                        swiglu_forward_pair_kernel<W, Weighted, true>
                            <<<(out.numel() + 511) / 512, 256, 0, stream>>>(
                                y.data_ptr<T>(), Weighted ? weights.data_ptr<W>() : nullptr,
                                out.data_ptr<T>(), h_divider, int(out.numel()));
                    } else {
                        swiglu_forward_pair_kernel<W, Weighted, false>
                            <<<(out.numel() + 511) / 512, 256, 0, stream>>>(
                                y.data_ptr<T>(), Weighted ? weights.data_ptr<W>() : nullptr,
                                out.data_ptr<T>(), h_divider, int(out.numel()));
                    }
                    C10_CUDA_KERNEL_LAUNCH_CHECK();
                    return;
                }
            }
            constexpr int Items = 4;
            if (h >= Items * kThreads) {
                const dim3 grid((h + Items * kThreads - 1) / (Items * kThreads),
                                std::min<int64_t>(rows, kGridRows),
                                (rows + kGridRows - 1) / kGridRows);
                swiglu_forward_scalar_kernel<T, W, Weighted, Bias, Items>
                    <<<grid, kThreads, 0, stream>>>(
                        y.data_ptr<T>(), Bias ? bias.data_ptr<T>() : nullptr,
                        Weighted ? weights.data_ptr<W>() : nullptr, out.data_ptr<T>(),
                        h, bias_stride, rows);
                C10_CUDA_KERNEL_LAUNCH_CHECK();
                return;
            }
        }
        // ACU identified dynamic row/tile division in forward; an explicit grid preserves tile order.
        // Encode rows in grid y/z to retain CUDA C support beyond 65535 rows.
        const dim3 grid(tiles, std::min<int64_t>(rows, kGridRows),
                        (rows + kGridRows - 1) / kGridRows);
        if constexpr (V > 1) {
            swiglu_forward_pack_kernel<T, W, V, Weighted, Bias>
                <<<grid, kThreads, 0, stream>>>(
                    y.data_ptr<T>(), Bias ? bias.data_ptr<T>() : nullptr,
                    Weighted ? weights.data_ptr<W>() : nullptr, out.data_ptr<T>(),
                    h, bias_stride, rows);
        } else {
            swiglu_kernel<T, W, V, Back, Weighted, Bias, kThreads, 0, true>
                <<<grid, kThreads, 0, stream>>>(
                    y.data_ptr<T>(), nullptr, Bias ? bias.data_ptr<T>() : nullptr,
                    Weighted ? weights.data_ptr<W>() : nullptr, out.data_ptr<T>(),
                    nullptr, nullptr, h, tiles, bias_stride, rows);
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return;
    }
    torch::Tensor partial;
    if constexpr (Back && Weighted) {
        if (tiles > 1) partial = torch::empty({rows, tiles}, y.options().dtype(torch::kFloat));
    }
    swiglu_kernel<T, W, V, Back, Weighted, Bias><<<rows * tiles, kThreads, 0, stream>>>(
        y.data_ptr<T>(), Back ? g.data_ptr<T>() : nullptr,
        Bias ? bias.data_ptr<T>() : nullptr,
        Weighted ? weights.data_ptr<W>() : nullptr, out.data_ptr<T>(),
        Back && Weighted ? dw.data_ptr<W>() : nullptr,
        partial.defined() ? partial.data_ptr<float>() : nullptr,
        h, tiles, bias_stride, rows);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if constexpr (Back && Weighted) {
        if (tiles > 1) {
            finish_weight_grad<W><<<(rows + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
                partial.data_ptr<float>(), dw.data_ptr<W>(), rows, tiles);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    }
}

template <typename T, typename W, bool Back, bool Weighted, bool Bias,
          int V = 16 / sizeof(T)>
void dispatch_pack(
    const torch::Tensor& y, const torch::Tensor& g, const torch::Tensor& bias,
    const torch::Tensor& weights, torch::Tensor& out, torch::Tensor& dw,
    int64_t rows, int64_t h, int64_t bias_stride, cudaStream_t stream) {
    // Align to the ISA access width on 890P, including gate/up offsets and every row start.
    constexpr int bytes = sizeof(T) * V;
    if (h % V == 0 && access_aligned(y, bytes, 2 * h * sizeof(T)) &&
        access_aligned(g, bytes, h * sizeof(T)) &&
        access_aligned(bias, bytes, bias_stride * sizeof(T)) &&
        access_aligned(out, bytes, (Back ? 2 : 1) * h * sizeof(T))) {
        launch<T, W, V, Back, Weighted, Bias>(
            y, g, bias, weights, out, dw, rows, h, bias_stride, stream);
    } else if constexpr (V > 1) {
        // Try 16/8/4/2-byte accesses before falling back to naturally aligned scalar accesses.
        dispatch_pack<T, W, Back, Weighted, Bias, V / 2>(
            y, g, bias, weights, out, dw, rows, h, bias_stride, stream);
    } else {
        TORCH_CHECK(false, "SwiGLU tensor is not naturally aligned");
    }
}

template <typename T, typename W, bool Weighted>
void dispatch_mode(
    const torch::Tensor& y, const torch::Tensor& g, const torch::Tensor& bias,
    const torch::Tensor& weights, torch::Tensor& out, torch::Tensor& dw,
    int64_t rows, int64_t h, int64_t bias_stride, cudaStream_t stream) {
    if (g.defined()) {
        if (bias.defined())
            dispatch_pack<T, W, true, Weighted, true>(
                y, g, bias, weights, out, dw, rows, h, bias_stride, stream);
        else
            dispatch_pack<T, W, true, Weighted, false>(
                y, g, bias, weights, out, dw, rows, h, bias_stride, stream);
    } else {
        if (bias.defined())
            dispatch_pack<T, W, false, Weighted, true>(
                y, g, bias, weights, out, dw, rows, h, bias_stride, stream);
        else
            dispatch_pack<T, W, false, Weighted, false>(
                y, g, bias, weights, out, dw, rows, h, bias_stride, stream);
    }
}

template <typename T>
void dispatch_dtype(
    const torch::Tensor& y, const torch::Tensor& g, const torch::Tensor& bias,
    const torch::Tensor& weights, torch::Tensor& out, torch::Tensor& dw,
    int64_t rows, int64_t h, int64_t bias_stride, cudaStream_t stream) {
    if (!weights.defined()) {
        dispatch_mode<T, float, false>(y, g, bias, weights, out, dw, rows, h, bias_stride, stream);
    } else if (weights.scalar_type() == y.scalar_type()) {
        dispatch_mode<T, T, true>(y, g, bias, weights, out, dw, rows, h, bias_stride, stream);
    } else {
        dispatch_mode<T, float, true>(y, g, bias, weights, out, dw, rows, h, bias_stride, stream);
    }
}

// Preserve original output shapes for non-tokenwise broadcasting; ATen moves data, CUDA C computes.
template <typename T, typename W, bool Back>
__global__ void weighted_broadcast_kernel(
    const T* a, const T* b, const T* g, const W* w, T* out,
    int64_t elements, int64_t h) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= elements) return;
    const float x = float(a[i]), z = float(b[i]);
    const float s = 1.f / (1.f + __expf(-x));
    if constexpr (Back) {
        const float scale = float(g[i]) * float(w[i]);
        const int64_t offset = (i / h) * (2 * h) + i % h;
        out[offset] = T(scale * s * (1.f + x * (1.f - s)) * z);
        out[offset + h] = T(scale * x * s);
    } else {
        out[i] = T((x * s * z) * float(w[i]));
    }
}

template <typename T, typename W>
__global__ void weighted_broadcast_dw_kernel(
    const T* a, const T* b, const T* g, W* dw, int64_t h) {
    const int64_t row = blockIdx.x;
    float value = 0.f;
    for (int64_t col = threadIdx.x; col < h; col += kThreads) {
        const int64_t i = row * h + col;
        const float x = float(a[i]);
        value += ((x / (1.f + __expf(-x))) * float(b[i])) * float(g[i]);
    }
    const float total = block_sum(value);
    if (threadIdx.x == 0) dw[row] = W(total);
}

// Read one-dimensional feature weights with their actual stride without materializing broadcasts.
template <typename T, typename W, int V, bool Pow2 = false>
__global__ __launch_bounds__(kThreads, 4) void weighted_feature_forward_kernel(
    const T* __restrict__ y, const W* __restrict__ weights, T* __restrict__ out,
    int64_t h, int64_t packs, int64_t weight_stride, int row_pack_shift = 0) {
    const int64_t index = int64_t(blockIdx.x) * kThreads + threadIdx.x;
    if (index >= packs) return;
    int64_t row, col;
    if constexpr (Pow2) {
        // The host verified a positive power-of-two Pack count; keep nonnegative 64-bit indices, even at shift=0.
        row = index >> row_pack_shift;
        col = (index & ((int64_t(1) << row_pack_shift) - 1)) * V;
    } else {
        const int64_t row_packs = h / V;
        row = index / row_packs;
        col = (index % row_packs) * V;
    }
    const int64_t offset = row * (2 * h) + col;
    const auto a = *reinterpret_cast<const Pack<T, V>*>(y + offset);
    const auto b = *reinterpret_cast<const Pack<T, V>*>(y + offset + h);
    Pack<T, V> result;
    #pragma unroll
    for (int j = 0; j < V; ++j) {
        const float x = float(a.data[j]);
        const float s = 1.f / (1.f + __expf(-x));
        result.data[j] = T(((x * s) * float(b.data[j])) *
                           float(weights[(col + j) * weight_stride]));
    }
    reinterpret_cast<Pack<T, V>*>(out)[index] = result;
}

// Share loads for dX and dW; split wide rows into fixed 1024-element tiles without atomic reductions.
template <typename T, typename W, int V, int RowThreads>
__global__ __launch_bounds__(kThreads, 4) void weighted_feature_backward_kernel(
    const T* __restrict__ y, const T* __restrict__ g,
    const W* __restrict__ weights, T* __restrict__ out, W* __restrict__ dw,
    std::conditional_t<std::is_same_v<W, float>, double, float>* __restrict__ partial,
    int64_t h, int64_t rows, int64_t tiles, int64_t weight_stride) {
    using Acc = std::conditional_t<std::is_same_v<W, float>, double, float>;
    constexpr int RowsPerBlock = kThreads / RowThreads;
    const int64_t row = RowThreads == 32
        ? int64_t(blockIdx.x) * RowsPerBlock + threadIdx.x / RowThreads
        : int64_t(blockIdx.y) + int64_t(blockIdx.z) * kGridRows;
    if (row >= rows) return;
    const int64_t tile = RowThreads == 32 ? 0 : int64_t(blockIdx.x);
    const int64_t begin = tile * (1024 / V), end = min(h / V, begin + 1024 / V);
    Acc sum = 0;
    for (int64_t v = begin + threadIdx.x % RowThreads; v < end; v += RowThreads) {
        const int64_t col = v * V, offset = row * (2 * h) + col;
        const auto a = *reinterpret_cast<const Pack<T, V>*>(y + offset);
        const auto b = *reinterpret_cast<const Pack<T, V>*>(y + offset + h);
        const auto grad = *reinterpret_cast<const Pack<T, V>*>(g + row * h + col);
        Pack<T, V> da, db;
        #pragma unroll
        for (int j = 0; j < V; ++j) {
            const float x = float(a.data[j]), z = float(b.data[j]);
            const float go = float(grad.data[j]);
            const float s = 1.f / (1.f + __expf(-x)), silu = x * s;
            const float scale = go * float(weights[(col + j) * weight_stride]);
            da.data[j] = T(scale * s * (1.f + x * (1.f - s)) * z);
            db.data[j] = T(scale * silu);
            sum += Acc((silu * z) * go);
        }
        *reinterpret_cast<Pack<T, V>*>(out + offset) = da;
        *reinterpret_cast<Pack<T, V>*>(out + offset + h) = db;
    }
    for (int delta = 16; delta; delta /= 2)
        sum += __shfl_down_sync(0xffffffff, sum, delta);
    if constexpr (RowThreads == kThreads) {
        __shared__ Acc warp_sums[kThreads / 32];
        const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
        if (lane == 0) warp_sums[warp] = sum;
        __syncthreads();
        sum = threadIdx.x < kThreads / 32 ? warp_sums[lane] : Acc(0);
        if (warp == 0) {
            for (int delta = 16; delta; delta /= 2)
                sum += __shfl_down_sync(0xffffffff, sum, delta);
        }
    }
    if (threadIdx.x % RowThreads == 0) {
        if (tiles == 1) dw[row] = W(sum);
        else partial[row * tiles + tile] = sum;
    }
}

template <typename W>
__global__ void weighted_feature_finish_kernel(
    const std::conditional_t<std::is_same_v<W, float>, double, float>* __restrict__ partial,
    W* __restrict__ dw, int64_t rows, int64_t tiles) {
    using Acc = std::conditional_t<std::is_same_v<W, float>, double, float>;
    const int64_t row = int64_t(blockIdx.x) * kThreads + threadIdx.x;
    if (row >= rows) return;
    Acc sum = 0;
    for (int64_t tile = 0; tile < tiles; ++tile) sum += partial[row * tiles + tile];
    dw[row] = W(sum);
}

template <typename T, typename W, int V = 16 / sizeof(T)>
void launch_weighted_feature(
    const torch::Tensor& y, const torch::Tensor& g, const torch::Tensor& weights,
    torch::Tensor& out, torch::Tensor& dw, int64_t h, int64_t rows,
    cudaStream_t stream) {
    constexpr int bytes = sizeof(T) * V;
    if (h % V == 0 && access_aligned(y, bytes, 2 * h * sizeof(T)) &&
        access_aligned(g, bytes, h * sizeof(T)) &&
        access_aligned(out, bytes, (g.defined() ? 2 : 1) * h * sizeof(T))) {
        if (!g.defined()) {
            const int64_t packs = out.numel() / V;
            if constexpr (std::is_same_v<T, float> && std::is_same_v<W, float> && V == 4) {
                const int64_t row_packs = h / V;
                if (row_packs > 0 && (row_packs & (row_packs - 1)) == 0) {
                    const int shift = __builtin_ctzll(static_cast<unsigned long long>(row_packs));
                    weighted_feature_forward_kernel<T, W, V, true>
                        <<<(packs + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
                            y.data_ptr<T>(), weights.data_ptr<W>(), out.data_ptr<T>(),
                            h, packs, weights.stride(0), shift);
                    C10_CUDA_KERNEL_LAUNCH_CHECK();
                    return;
                }
            }
            weighted_feature_forward_kernel<T, W, V>
                <<<(packs + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
                    y.data_ptr<T>(), weights.data_ptr<W>(), out.data_ptr<T>(),
                    h, packs, weights.stride(0));
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            return;
        }
        using Acc = std::conditional_t<std::is_same_v<W, float>, double, float>;
        const int64_t tiles = (h + 1023) / 1024;
        torch::Tensor partial;
        if (tiles > 1)
            partial = torch::empty({rows, tiles}, y.options().dtype(
                std::is_same_v<Acc, double> ? torch::kDouble : torch::kFloat));
        if (h <= 128) {
            weighted_feature_backward_kernel<T, W, V, 32>
                <<<(rows + 3) / 4, kThreads, 0, stream>>>(
                    y.data_ptr<T>(), g.data_ptr<T>(), weights.data_ptr<W>(),
                    out.data_ptr<T>(), dw.data_ptr<W>(), nullptr,
                    h, rows, 1, weights.stride(0));
        } else {
            const dim3 grid(tiles, std::min<int64_t>(rows, kGridRows),
                            (rows + kGridRows - 1) / kGridRows);
            weighted_feature_backward_kernel<T, W, V, kThreads>
                <<<grid, kThreads, 0, stream>>>(
                    y.data_ptr<T>(), g.data_ptr<T>(), weights.data_ptr<W>(),
                    out.data_ptr<T>(), dw.data_ptr<W>(),
                    partial.defined() ? partial.data_ptr<Acc>() : nullptr,
                    h, rows, tiles, weights.stride(0));
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        if (tiles > 1) {
            weighted_feature_finish_kernel<W>
                <<<(rows + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
                    partial.data_ptr<Acc>(), dw.data_ptr<W>(), rows, tiles);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    } else if constexpr (V > 1) {
        launch_weighted_feature<T, W, V / 2>(y, g, weights, out, dw, h, rows, stream);
    } else {
        TORCH_CHECK(false, "SwiGLU tensor is not naturally aligned");
    }
}

template <typename T, typename W>
std::vector<torch::Tensor> launch_weighted_broadcast(
    const torch::Tensor& y, const torch::Tensor& g, const torch::Tensor& weights,
    cudaStream_t stream) {
    const int64_t h = y.size(-1) / 2;
    auto feature_shape = y.sizes().vec();
    feature_shape.back() = h;
    if (weights.dim() == 1 && weights.size(0) == h &&
        (!g.defined() || g.sizes().vec() == feature_shape)) {
        const int64_t elements = y.numel() / 2, rows = elements / h;
        TORCH_CHECK((elements + kThreads - 1) / kThreads <= std::numeric_limits<int32_t>::max(),
                    "SwiGLU launch grid is too large");
        auto out = torch::empty(g.defined() ? y.sizes().vec() : feature_shape, y.options());
        torch::Tensor dw;
        if (g.defined()) {
            TORCH_CHECK(rows <= std::numeric_limits<int32_t>::max(), "SwiGLU launch grid is too large");
            feature_shape.back() = 1;
            dw = torch::empty(feature_shape, weights.options());
        }
        if (rows) {
            const auto dense_g = g.defined() ? g.contiguous() : torch::Tensor();
            launch_weighted_feature<T, W>(y, dense_g, weights, out, dw, h, rows, stream);
        }
        return g.defined() ? std::vector<torch::Tensor>{out, dw} : std::vector<torch::Tensor>{out};
    }
    const auto a = y.slice(-1, 0, h), b = y.slice(-1, h, 2 * h);
    auto parts = g.defined() ? at::broadcast_tensors({a, b, g, weights})
                             : at::broadcast_tensors({a, b, weights});
    auto shape = parts[0].sizes().vec();
    const int64_t width = shape.back(), elements = parts[0].numel();
    for (auto& part : parts) part = part.contiguous();
    if (g.defined()) shape.back() *= 2;
    auto out = torch::empty(shape, y.options());
    const int64_t blocks = (elements + kThreads - 1) / kThreads;
    TORCH_CHECK(blocks <= std::numeric_limits<int32_t>::max(), "SwiGLU launch grid is too large");
    if (elements) {
        if (g.defined()) {
            weighted_broadcast_kernel<T, W, true><<<blocks, kThreads, 0, stream>>>(
                parts[0].data_ptr<T>(), parts[1].data_ptr<T>(), parts[2].data_ptr<T>(),
                parts[3].data_ptr<W>(), out.data_ptr<T>(), elements, width);
        } else {
            weighted_broadcast_kernel<T, W, false><<<blocks, kThreads, 0, stream>>>(
                parts[0].data_ptr<T>(), parts[1].data_ptr<T>(), nullptr,
                parts[2].data_ptr<W>(), out.data_ptr<T>(), elements, width);
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    if (!g.defined()) return {out};
    // Original dW depends only on swiglu(y) and g; broadcasting weights must not expand its shape.
    auto reduced = at::broadcast_tensors({a, b, g});
    auto dw_shape = reduced[0].sizes().vec();
    const int64_t dw_h = dw_shape.back();
    dw_shape.back() = 1;
    auto dw = torch::empty(dw_shape, weights.options());
    TORCH_CHECK(dw.numel() <= std::numeric_limits<int32_t>::max(), "SwiGLU launch grid is too large");
    for (auto& part : reduced) part = part.contiguous();
    if (dw.numel()) {
        weighted_broadcast_dw_kernel<T, W><<<dw.numel(), kThreads, 0, stream>>>(
            reduced[0].data_ptr<T>(), reduced[1].data_ptr<T>(), reduced[2].data_ptr<T>(),
            dw.data_ptr<W>(), dw_h);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {out, dw};
}

void check_peer(const torch::Tensor& t, const torch::Tensor& y, const char* name) {
    TORCH_CHECK(t.device() == y.device(), name, " must be on the input device");
    TORCH_CHECK(t.layout() == torch::kStrided, name, " must have strided layout");
}
}  // namespace

std::vector<torch::Tensor> swiglu_cuda(
    torch::Tensor input, std::optional<torch::Tensor> grad,
    std::optional<torch::Tensor> bias_arg, std::optional<torch::Tensor> weights_arg) {
    TORCH_CHECK(input.is_cuda(), "SwiGLU CUDA C requires a CUDA/HGGC tensor");
    TORCH_CHECK(input.layout() == torch::kStrided, "input must have strided layout");
    TORCH_CHECK(input.dim() >= 1 && input.size(-1) > 0 && input.size(-1) % 2 == 0,
                "input last dimension must be positive and even");
    TORCH_CHECK(input.scalar_type() == torch::kHalf ||
                input.scalar_type() == torch::kBFloat16 || input.scalar_type() == torch::kFloat,
                "input dtype must be float16, bfloat16 or float32");
    const c10::cuda::CUDAGuard guard(input.device());
    const auto* properties = at::cuda::getDeviceProperties(input.get_device());
    TORCH_CHECK(properties->major == 8 && properties->minor == 9,
                "SwiGLU CUDA C requires sm89, got sm", properties->major, properties->minor);
    auto y = input.contiguous();
    const int64_t h = y.size(-1) / 2;
    const int64_t rows = y.numel() / (2 * h);
    auto output_shape = y.sizes().vec();
    output_shape.back() = h;
    torch::Tensor g, bias, weights, dw;
    if (weights_arg.has_value()) {
        check_peer(*weights_arg, y, "weights");
        TORCH_CHECK(weights_arg->scalar_type() == y.scalar_type() ||
                    weights_arg->scalar_type() == torch::kFloat,
                    "weights dtype must match input or be float32");
        if (grad.has_value()) {
            check_peer(*grad, y, "g");
            TORCH_CHECK(grad->scalar_type() == y.scalar_type(), "g dtype must match input");
        }
        const auto broadcast_shape = at::infer_size(output_shape, weights_arg->sizes());
        const bool token_weights = (weights_arg->dim() == 0 || weights_arg->size(-1) == 1) &&
                                   broadcast_shape == output_shape;
        if (!token_weights || (grad.has_value() && grad->sizes().vec() != output_shape)) {
            TORCH_CHECK(!bias_arg.has_value(), "Bias is not supported for weighted swiglu fusion");
            std::vector<torch::Tensor> result;
            const auto stream = at::cuda::getCurrentCUDAStream(y.get_device());
            AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
                y.scalar_type(), "weighted_swiglu_broadcast", [&] {
                    if (weights_arg->scalar_type() == y.scalar_type())
                        result = launch_weighted_broadcast<scalar_t, scalar_t>(
                            y, grad.value_or(torch::Tensor()), *weights_arg, stream);
                    else
                        result = launch_weighted_broadcast<scalar_t, float>(
                            y, grad.value_or(torch::Tensor()), *weights_arg, stream);
                });
            return result;
        }
    }
    int64_t bias_stride = 0;
    if (grad.has_value()) {
        check_peer(*grad, y, "g");
        TORCH_CHECK(grad->scalar_type() == y.scalar_type(), "g dtype must match input");
        TORCH_CHECK(grad->sizes().vec() == output_shape, "g shape must match SwiGLU output");
        g = grad->contiguous();
    }
    if (bias_arg.has_value()) {
        check_peer(*bias_arg, y, "bias");
        TORCH_CHECK(bias_arg->scalar_type() == y.scalar_type(), "bias dtype must match input");
        auto expanded = bias_arg->expand(y.sizes());
        if (bias_arg->numel() == 2 * h && bias_arg->size(-1) == 2 * h) {
            bias = bias_arg->contiguous();
        } else {
            bias = expanded.contiguous();
            bias_stride = 2 * h;
        }
    }
    if (weights_arg.has_value()) {
        check_peer(*weights_arg, y, "weights");
        TORCH_CHECK(weights_arg->scalar_type() == y.scalar_type() ||
                    weights_arg->scalar_type() == torch::kFloat,
                    "weights dtype must match input or be float32");
        auto weight_shape = output_shape;
        weight_shape.back() = 1;
        weights = weights_arg->expand(weight_shape).contiguous();
        if (g.defined()) dw = torch::empty(weight_shape, weights.options());
    }
    auto out = torch::empty(g.defined() ? y.sizes().vec() : output_shape, y.options());
    if (rows != 0) {
        const auto stream = at::cuda::getCurrentCUDAStream(y.get_device());
        AT_DISPATCH_SWITCH(y.scalar_type(), "swiglu_cuda",
            AT_DISPATCH_CASE(at::ScalarType::Half, [&] {
                dispatch_dtype<scalar_t>(y, g, bias, weights, out, dw, rows, h, bias_stride, stream);
            })
            AT_DISPATCH_CASE(at::ScalarType::BFloat16, [&] {
                dispatch_dtype<scalar_t>(y, g, bias, weights, out, dw, rows, h, bias_stride, stream);
            })
            AT_DISPATCH_CASE(at::ScalarType::Float, [&] {
                dispatch_dtype<scalar_t>(y, g, bias, weights, out, dw, rows, h, bias_stride, stream);
            })
        );
    }
    if (g.defined() && weights.defined()) {
        // Match the original low-level API: reduce only the last axis; autograd handles broadcast reduction.
        return {out, dw};
    }
    return {out};
}
