// Metal Shading Language kernels for the GPU Flash Attention backend.
//
// Layout: Q/K/V/O are contiguous [batch, heads, seq, dim] fp32 tensors.
// `LANE_DIM` and `QUERY_LANES` are injected by the Rust side at
// pipeline-compile time: `QUERY_LANES` is 4 for dim <= 64 and 8 for larger
// dims, and `LANE_DIM = dim / QUERY_LANES`. One lane group owns one query row,
// one lane covers LANE_DIM adjacent channels.
//
// Flash Attention kernel strategy (4 or 8 lanes per query):
//   * each lane group owns one query row, holding q and the output accumulator
//     in registers (LANE_DIM floats per lane),
//   * the threadgroup cooperatively stages a [block_cols, dim] K/V tile in
//     threadgroup memory,
//   * for every key the lanes compute a partial dot product, reduce it with
//     2–3 `simd_shuffle_xor` steps, then update the online softmax state
//     (m, l) redundantly in all lanes,
//   * 8-key sub-tiles keep scores in registers so the exponentials within a
//     sub-tile are independent,
//   * the causal mask is implemented as score = -inf, never as control-flow
//     divergence: every SIMD-group operation must be executed by all lanes.
//
// A naive materialized-score path (scores -> row softmax -> P·V) is also
// provided as a baseline that exposes the O(seq^2) memory cost.

#include <metal_stdlib>
using namespace metal;

#ifndef LANE_DIM
#define LANE_DIM 8
#endif
#ifndef QUERY_LANES
#define QUERY_LANES 8
#endif

struct AttnParams {
    uint batch;
    uint heads;
    uint seq;
    uint dim;
    float scale;
    uint causal;
    uint block_cols;
    uint block_rows;
};

constant constexpr uint kSub = QUERY_LANES;
// Keys processed per register sub-tile (must divide block_cols).
constant constexpr uint SUB_TILE = 8;

// ---------------------------------------------------------------------------
// Flash Attention: tiled online softmax, no seq x seq materialization.
// ---------------------------------------------------------------------------
kernel void flash_attention_kernel(
    device const float* Q [[buffer(0)]],
    device const float* K [[buffer(1)]],
    device const float* V [[buffer(2)]],
    device float* O [[buffer(3)]],
    constant AttnParams& p [[buffer(4)]],
    threadgroup float* smem [[threadgroup(0)]],
    uint3 tid3 [[thread_position_in_threadgroup]],
    uint3 tgid3 [[threadgroup_position_in_grid]],
    uint3 tg_size3 [[threads_per_threadgroup]])
{
    const uint tid = tid3.x;
    const uint tgid_x = tgid3.x;
    const uint bh = tgid3.y;
    const uint tg_size = tg_size3.x;
    const uint dim = p.dim;
    const uint seq = p.seq;
    const uint bc = p.block_cols;
    const uint br = p.block_rows;

    const uint lane = tid & (kSub - 1);
    const uint qloc = tid / kSub;
    const uint qi = tgid_x * br + qloc;

    const uint head_stride = seq * dim;
    const uint base = bh * head_stride;
    const bool active = qi < seq;
    // Inactive lanes still run the whole loop (SIMD-group width must stay
    // uniform); they compute a duplicate of the last valid query and skip the
    // final store.
    const uint q_eff = active ? qi : (seq - 1);

    float qreg[LANE_DIM];
    float oreg[LANE_DIM];
#pragma unroll
    for (uint t = 0; t < LANE_DIM; ++t) {
        qreg[t] = 0.0f;
        oreg[t] = 0.0f;
    }

    const uint d0 = lane * LANE_DIM;
    {
        const device float* qp = Q + base + q_eff * dim + d0;
#pragma unroll
        for (uint t = 0; t < LANE_DIM; ++t) {
            qreg[t] = qp[t];
        }
    }

    // The causal variant of a query block never needs keys beyond the largest
    // query of that block.
    const uint key_limit = p.causal ? min(seq, (tgid_x + 1) * br) : seq;

    float m = -INFINITY;
    float l = 0.0f;

    threadgroup float* ks = smem;
    threadgroup float* vs = smem + bc * dim;

    for (uint j0 = 0; j0 < key_limit; j0 += bc) {
        const uint j1 = min(j0 + bc, key_limit);
        const uint jn = j1 - j0;

        // Cooperative staging of the current K/V tile. Zero-fill the tail so
        // the compute loop can stay branch free.
        const uint nvec = bc * (dim / 4);
        for (uint idx = tid; idx < nvec; idx += tg_size) {
            const uint j = (idx * 4) / dim;
            const uint d = (idx * 4) % dim;
            const uint jj = j0 + j;
            float4 kk = float4(0.0f);
            float4 vv = float4(0.0f);
            if (jj < j1) {
                kk = *(const device float4*)(K + base + jj * dim + d);
                vv = *(const device float4*)(V + base + jj * dim + d);
            }
            ((threadgroup float4*)ks)[idx] = kk;
            ((threadgroup float4*)vs)[idx] = vv;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Process the staged tile in sub-tiles of SUB_TILE keys. Scores are
        // kept in registers so each sub-tile only rescales the running state
        // once, and all exponentials in a sub-tile are independent (no
        // loop-carried exp chain across keys).
        for (uint js = 0; js < jn; js += SUB_TILE) {
            float sreg[SUB_TILE];
#pragma unroll
            for (uint t = 0; t < SUB_TILE; ++t) {
                const uint j = js + t;
                float score = -INFINITY;
                if (j < jn) {
                    float acc0 = 0.0f;
                    float acc1 = 0.0f;
                    float acc2 = 0.0f;
                    float acc3 = 0.0f;
#if (LANE_DIM % 4) == 0
                    const threadgroup float4* kp =
                        (const threadgroup float4*)(ks + j * dim + d0);
#pragma unroll
                    for (uint q4 = 0; q4 < LANE_DIM / 4; ++q4) {
                        const float4 kk = kp[q4];
                        acc0 += qreg[4 * q4 + 0] * kk.x;
                        acc1 += qreg[4 * q4 + 1] * kk.y;
                        acc2 += qreg[4 * q4 + 2] * kk.z;
                        acc3 += qreg[4 * q4 + 3] * kk.w;
                    }
#else
#pragma unroll
                    for (uint q4 = 0; q4 < LANE_DIM; ++q4) {
                        acc0 += qreg[q4] * ks[j * dim + d0 + q4];
                    }
#endif
                    score = (acc0 + acc1) + (acc2 + acc3);
                    // Reduce across the query's lane group; the xor masks stay
                    // inside the subwarp, so no cross-query traffic.
#if QUERY_LANES >= 2
                    score += simd_shuffle_xor(score, ushort(1));
#endif
#if QUERY_LANES >= 4
                    score += simd_shuffle_xor(score, ushort(2));
#endif
#if QUERY_LANES >= 8
                    score += simd_shuffle_xor(score, ushort(4));
#endif
#if QUERY_LANES >= 16
                    score += simd_shuffle_xor(score, ushort(8));
#endif
                    score *= p.scale;
                    if (p.causal && (j0 + j) > q_eff) {
                        score = -INFINITY;
                    }
                }
                sreg[t] = score;
            }

            float block_m = -INFINITY;
#pragma unroll
            for (uint t = 0; t < SUB_TILE; ++t) {
                block_m = max(block_m, sreg[t]);
            }

            const float new_m = max(m, block_m);
            const float alpha = (m == -INFINITY) ? 0.0f : fast::exp(m - new_m);
            if (alpha != 1.0f) {
                l *= alpha;
#pragma unroll
                for (uint t = 0; t < LANE_DIM; ++t) {
                    oreg[t] *= alpha;
                }
            }

#pragma unroll
            for (uint t = 0; t < SUB_TILE; ++t) {
                const float pw = fast::exp(sreg[t] - new_m);
                l += pw;
                const uint j = js + t;
#if (LANE_DIM % 4) == 0
                const threadgroup float4* vp =
                    (const threadgroup float4*)(vs + j * dim + d0);
#pragma unroll
                for (uint q4 = 0; q4 < LANE_DIM / 4; ++q4) {
                    const float4 vv = vp[q4];
                    oreg[4 * q4 + 0] += pw * vv.x;
                    oreg[4 * q4 + 1] += pw * vv.y;
                    oreg[4 * q4 + 2] += pw * vv.z;
                    oreg[4 * q4 + 3] += pw * vv.w;
                }
#else
#pragma unroll
                for (uint q4 = 0; q4 < LANE_DIM; ++q4) {
                    oreg[q4] += pw * vs[j * dim + d0 + q4];
                }
#endif
            }
            m = new_m;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (active) {
        device float* op = O + base + qi * dim + d0;
        const float inv_l = 1.0f / l;
#pragma unroll
        for (uint t = 0; t < LANE_DIM; ++t) {
            op[t] = oreg[t] * inv_l;
        }
    }
}

// ---------------------------------------------------------------------------
// Naive baseline, stage 1: materialize the full [seq, seq] score matrix.
// ---------------------------------------------------------------------------
kernel void scores_kernel(
    device const float* Q [[buffer(0)]],
    device const float* K [[buffer(1)]],
    device float* S [[buffer(2)]],
    constant AttnParams& p [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]])
{
    const uint j = gid.x;
    const uint i = gid.y;
    const uint bh = gid.z;
    if (i >= p.seq || j >= p.seq) {
        return;
    }

    const uint base = bh * p.seq * p.dim;
    const device float* q = Q + base + i * p.dim;
    const device float* k = K + base + j * p.dim;

    float acc = 0.0f;
    for (uint d = 0; d < p.dim; ++d) {
        acc += q[d] * k[d];
    }

    float s = acc * p.scale;
    if (p.causal && j > i) {
        s = -INFINITY;
    }
    S[(bh * p.seq + i) * p.seq + j] = s;
}

// Stage 2: row-wise softmax in place (three streaming passes).
kernel void softmax_rows_kernel(
    device float* S [[buffer(0)]],
    constant AttnParams& p [[buffer(1)]],
    uint row [[thread_position_in_grid]])
{
    const uint total = p.batch * p.heads * p.seq;
    if (row >= total) {
        return;
    }

    device float* r = S + row * p.seq;
    float mx = -INFINITY;
    for (uint j = 0; j < p.seq; ++j) {
        mx = max(mx, r[j]);
    }

    float sum = 0.0f;
    for (uint j = 0; j < p.seq; ++j) {
        const float e = exp(r[j] - mx);
        r[j] = e;
        sum += e;
    }

    const float inv = 1.0f / sum;
    for (uint j = 0; j < p.seq; ++j) {
        r[j] *= inv;
    }
}

// Stage 3: O = P · V.
kernel void pv_kernel(
    device const float* P [[buffer(0)]],
    device const float* V [[buffer(1)]],
    device float* O [[buffer(2)]],
    constant AttnParams& p [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]])
{
    const uint d = gid.x;
    const uint i = gid.y;
    const uint bh = gid.z;
    if (i >= p.seq || d >= p.dim) {
        return;
    }

    const uint head_stride = p.seq * p.dim;
    const uint base = bh * head_stride;
    const device float* prob = P + (bh * p.seq + i) * p.seq;

    float acc = 0.0f;
    for (uint j = 0; j < p.seq; ++j) {
        acc += prob[j] * V[base + j * p.dim + d];
    }
    O[base + i * p.dim + d] = acc;
}
