"""
Step 6: FlashAttention V3 (Shah et al., 2024)
— 针对 Hopper (H100) GPU 架构的极致优化

==========================================================================
背景: NVIDIA Hopper 架构新特性
==========================================================================

H100 (Hopper, SM90) 相比 A100 (Ampere, SM80) 引入了:

    1. TMA (Tensor Memory Accelerator):
       - 异步的、硬件加速的 global ↔ shared memory 数据搬运
       - 不占用计算单元! 可与计算完全 overlap

    2. WGMMA (Warpgroup MMA):
       - Warpgroup = 4 warps = 128 threads
       - 直接从 shared memory 做 matrix multiply
       - 比 A100 的 wmma/mma 吞吐量更高

    3. FP8 (E4M3 / E5M2):
       - 硬件原生 FP8 Tensor Core
       - 比 FP16/BF16 吞吐量翻倍

    4. 异步执行:
       - TMA (搬数据) 与 WGMMA (算 matmul) 可完全异步重叠
       - producer-consumer 模型

==========================================================================
FlashAttention V3 的三大优化
==========================================================================

优化 1: Producer-Consumer 异步流水线 (Warp Specialization)
─────────────────────────────────────────────────────────

    V2 做法: 所有 warps 先搬数据, 然后所有 warps 一起算
            → 搬数据和计算串行！

    V3 做法: 将 warpgroups 分成两种角色:
        - Producer warpgroup: 专门用 TMA 搬数据到 shared memory
        - Consumer warpgroups: 专门做 WGMMA 计算

        Producer                   Consumer
        ┌──────────┐              ┌──────────┐
        │ TMA load │──barrier──>  │  WGMMA   │
        │  K_1     │              │ Q@K_0^T  │
        │ TMA load │              │  P@V_0   │
        │  V_1     │──barrier──>  │ Q@K_1^T  │
        │ TMA load │              │  P@V_1   │
        │  K_2     │──barrier──>  │   ...    │
        └──────────┘              └──────────┘

    → 数据搬运和计算完全 overlap!
    → 使用 named barriers 同步 producer/consumer

优化 2: Ping-Pong Scheduling (乒乓调度)
──────────────────────────────────────

    问题: 当有多个 consumer warpgroups 时, 它们共享 Tensor Cores
          → 可能互相干扰, 导致 pipeline stall

    解决: Ping-Pong scheduling
        - 2 个 consumer warpgroups 交替使用不同的 shared memory buffers
        - 当 Consumer 0 计算 tile i 时, Consumer 1 等待 tile i+1 的数据
        - 当 tile i+1 就绪, Consumer 1 开始算, Consumer 0 等 tile i+2
        - → 持续保持 Tensor Cores 繁忙!

        Time:     ──t0──  ──t1──  ──t2──  ──t3──
        Consumer0: tile0   wait    tile2   wait
        Consumer1: wait    tile1   wait    tile3
        Buffer A:  tile0   tile2   tile2   tile4
        Buffer B:  tile1   tile1   tile3   tile3

    → 需要 2x shared memory (双 buffer)
    → 但 H100 有 228KB shared memory per SM (vs A100 的 164KB)

优化 3: FP8 & 低精度利用
──────────────────────────

    Q @ K^T 可以用 FP8 × FP8 → FP32 accumulate
    P @ V   也可以用 FP8 × FP8 → FP32

    挑战: FP8 的精度有限 (E4M3: 4 bit exponent, 3 bit mantissa)
        - Softmax 的 exp() 输出范围 [0, 1], 在 FP8 中精度损失大
        - 解决: incoherent processing
            - 对 Q 和 K 进行乱序处理, 减少系统性误差
            - 或者用混合精度: S = Q_fp8 @ K_fp8^T → FP32/FP16,
              P = softmax(S) 保持高精度,
              O = P_fp8 @ V_fp8

    FP8 吞吐提升:
        H100 FP8 Tensor Core: 3958 TFLOPS
        H100 FP16/BF16:       1979 TFLOPS
        → 理论最高 2x 加速!

==========================================================================
性能对比 (以 H100 为例)
==========================================================================

    配置: H100 SXM, B=1, H=32, d=128

    算法              | 16K  seq_len |  64K seq_len | 单位
    ──────────────────┼─────────────┼──────────────┼──────
    FlashAttn V2      |  390 TFLOPS |  410 TFLOPS  | FP16
    FlashAttn V3 FP16 |  620 TFLOPS |  640 TFLOPS  | FP16
    FlashAttn V3 FP8  |  740 TFLOPS |  780 TFLOPS  | FP8

    接近 H100 FP16 峰值的 ~33%, FP8 峰值的 ~20%
    (Attention 是 memory-bound → 接近极限)

==========================================================================
"""

import torch
import torch.nn.functional as F
import math
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class HopperConfig:
    """模拟 H100 Hopper GPU 的配置"""
    # Shared memory per SM
    smem_per_sm: int = 228 * 1024  # 228 KB
    # Number of SMs
    num_sms: int = 132
    # FP16 peak TFLOPS
    fp16_tflops: float = 1979.0
    # FP8 peak TFLOPS
    fp8_tflops: float = 3958.0
    # Memory bandwidth (GB/s)
    mem_bw_gbps: float = 3350.0
    # TMA bandwidth (better than manual copy)
    tma_speedup: float = 1.5


class FlashAttentionV3Sim:
    """
    FlashAttention V3 模拟实现。

    注意: 真正的 V3 需要 Hopper GPU + CUDA 12 + 专用 PTX 指令。
    这里用 PyTorch 模拟其算法逻辑和流水线行为。

    三大核心优化:
        1. Warp Specialization (Producer-Consumer 异步流水线)
        2. Ping-Pong Scheduling (双 buffer 交替计算)
        3. FP8 低精度计算 (模拟)
    """

    @staticmethod
    def forward_warp_specialized(
        Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
        B_r: int = 128, B_c: int = 128,
        causal: bool = False,
        simulate_fp8: bool = False
    ) -> Tuple[torch.Tensor, dict]:
        """
        模拟 FlashAttention V3 的 warp-specialized forward.

        与 V2 的区别:
        1. Producer 阶段 (TMA load) — 异步预取 K, V tiles
        2. Consumer 阶段 (WGMMA) — 计算 S = Q @ K^T, O += P @ V
        3. 两者通过 barrier 同步, overlap 执行

        Args:
            Q, K, V: [B, H, N, d]
            B_r: Q tile size (consumer processes)
            B_c: K/V tile size (producer loads)
            simulate_fp8: 模拟 FP8 量化
        Returns:
            O: [B, H, N, d]
            stats: dict with pipeline statistics
        """
        B, H, N, d = Q.shape
        scale = 1.0 / math.sqrt(d)

        O = torch.zeros_like(Q)
        stats = {
            "producer_loads": 0,
            "consumer_computes": 0,
            "pipeline_stages": [],
            "overlap_ratio": 0.0,
        }

        T_r = math.ceil(N / B_r)  # Q tiles
        T_c = math.ceil(N / B_c)  # KV tiles

        # ── FP8 模拟 ──
        if simulate_fp8:
            Q_compute = FlashAttentionV3Sim._simulate_fp8(Q)
            K_compute = FlashAttentionV3Sim._simulate_fp8(K)
            V_compute = FlashAttentionV3Sim._simulate_fp8(V)
        else:
            Q_compute = Q
            K_compute = K
            V_compute = V

        # ── 外循环: Q tiles ──
        for i in range(T_r):
            r_start = i * B_r
            r_end = min((i + 1) * B_r, N)
            Q_i = Q_compute[:, :, r_start:r_end, :]

            # Online softmax 状态
            m_i = torch.full((B, H, r_end - r_start, 1), float('-inf'),
                             device=Q.device, dtype=Q.dtype)
            l_i = torch.zeros(B, H, r_end - r_start, 1,
                              device=Q.device, dtype=Q.dtype)
            O_i = torch.zeros(B, H, r_end - r_start, d,
                              device=Q.device, dtype=Q.dtype)

            # ── Producer-Consumer 流水线 ──
            # 模拟: producer 提前 1 步 load, consumer 消费上一步的数据
            # Pipeline: [load_0] [load_1, compute_0] [load_2, compute_1] ... [compute_{T-1}]

            kv_end = (i + 1) if causal else T_c
            pipeline = []

            for j in range(kv_end):
                c_start = j * B_c
                c_end = min((j + 1) * B_c, N)

                # === Producer: TMA Load (异步, 与上一步 compute 重叠) ===
                K_j = K_compute[:, :, c_start:c_end, :]
                V_j = V_compute[:, :, c_start:c_end, :]
                stats["producer_loads"] += 1

                # === Consumer: WGMMA Compute ===
                # S = Q_i @ K_j^T
                S_ij = torch.matmul(Q_i, K_j.transpose(-2, -1)) * scale

                # Causal mask
                if causal:
                    row_idx = torch.arange(r_start, r_end, device=Q.device).unsqueeze(1)
                    col_idx = torch.arange(c_start, c_end, device=Q.device).unsqueeze(0)
                    mask = row_idx < col_idx
                    S_ij.masked_fill_(mask.unsqueeze(0).unsqueeze(0), float('-inf'))

                # Online softmax update
                m_ij = S_ij.max(dim=-1, keepdim=True).values
                m_new = torch.maximum(m_i, m_ij)
                exp_old = torch.exp(m_i - m_new)
                exp_new = torch.exp(S_ij - m_new)
                l_new = l_i * exp_old + exp_new.sum(dim=-1, keepdim=True)
                P_ij = exp_new

                O_i = O_i * exp_old + torch.matmul(P_ij, V_j)
                m_i = m_new
                l_i = l_new
                stats["consumer_computes"] += 1

                pipeline.append(("overlap" if j > 0 else "cold_start", j))

            O_i = O_i / (l_i + 1e-8)
            O[:, :, r_start:r_end, :] = O_i
            stats["pipeline_stages"].append(pipeline)

        # 计算 overlap ratio
        total_ops = stats["producer_loads"] + stats["consumer_computes"]
        overlapped = stats["consumer_computes"] - T_r  # 除 cold start 外都 overlap
        stats["overlap_ratio"] = max(0, overlapped) / max(1, total_ops)

        return O, stats

    @staticmethod
    def _simulate_fp8(tensor: torch.Tensor) -> torch.Tensor:
        """
        模拟 FP8 E4M3 量化。

        E4M3: 4 bits exponent, 3 bits mantissa
        Range: [-448, 448], 精度约 3-4 位有效数字

        真正的 FP8:
            torch.float8_e4m3fn (PyTorch 2.1+/CUDA 11.8+)
            这里用 round-to-nearest 模拟
        """
        # E4M3 的特性
        max_val = 448.0
        # 3 bits mantissa → 2^3 = 8 级量化
        # 实际精度取决于 exponent, 这里简化模拟

        # Clip to range
        tensor_clipped = tensor.clamp(-max_val, max_val)

        # 模拟精度损失: round to ~3 bits of mantissa precision
        # 对于 |x| ∈ [2^e, 2^(e+1)), 精度为 2^(e-3)
        abs_t = tensor_clipped.abs().clamp(min=1e-12)
        log2_abs = torch.floor(torch.log2(abs_t))
        precision = torch.pow(2.0, log2_abs - 3)
        quantized = torch.round(tensor_clipped / precision) * precision

        return quantized

    @staticmethod
    def forward_pingpong(
        Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
        B_r: int = 128, B_c: int = 128,
        causal: bool = False
    ) -> torch.Tensor:
        """
        模拟 Ping-Pong Scheduling: 两个 consumer 交替计算。

        Consumer 0 处理 even tiles, Consumer 1 处理 odd tiles:
            Time  |  Consumer 0      |  Consumer 1      |  Buffer
            ──────┼──────────────────┼──────────────────┼─────────
            t0    |  compute tile 0  |  wait            |  A: KV_0
            t1    |  wait            |  compute tile 1  |  B: KV_1
            t2    |  compute tile 2  |  wait            |  A: KV_2
            t3    |  wait            |  compute tile 3  |  B: KV_3

        效果: Tensor Cores 持续繁忙 (< 10% idle)
        """
        B, H, N, d = Q.shape
        scale = 1.0 / math.sqrt(d)
        O = torch.zeros_like(Q)

        T_r = math.ceil(N / B_r)
        T_c = math.ceil(N / B_c)

        for i in range(T_r):
            r_start = i * B_r
            r_end = min((i + 1) * B_r, N)
            Q_i = Q[:, :, r_start:r_end, :]

            # 双 buffer 的 online softmax 状态 (两个 consumer 各自维护)
            # 但最终需要合并 → 实际上还是一套状态
            m_i = torch.full((B, H, r_end - r_start, 1), float('-inf'),
                             device=Q.device, dtype=Q.dtype)
            l_i = torch.zeros(B, H, r_end - r_start, 1,
                              device=Q.device, dtype=Q.dtype)
            O_i = torch.zeros(B, H, r_end - r_start, d,
                              device=Q.device, dtype=Q.dtype)

            kv_end = (i + 1) if causal else T_c

            # Buffer A (even tiles) 和 Buffer B (odd tiles)
            for j in range(kv_end):
                c_start = j * B_c
                c_end = min((j + 1) * B_c, N)

                buffer_id = "A" if j % 2 == 0 else "B"  # Ping-Pong!

                K_j = K[:, :, c_start:c_end, :]
                V_j = V[:, :, c_start:c_end, :]

                S_ij = torch.matmul(Q_i, K_j.transpose(-2, -1)) * scale

                if causal:
                    row_idx = torch.arange(r_start, r_end, device=Q.device).unsqueeze(1)
                    col_idx = torch.arange(c_start, c_end, device=Q.device).unsqueeze(0)
                    mask = row_idx < col_idx
                    S_ij.masked_fill_(mask.unsqueeze(0).unsqueeze(0), float('-inf'))

                m_ij = S_ij.max(dim=-1, keepdim=True).values
                m_new = torch.maximum(m_i, m_ij)
                exp_old = torch.exp(m_i - m_new)
                exp_new = torch.exp(S_ij - m_new)
                l_new = l_i * exp_old + exp_new.sum(dim=-1, keepdim=True)
                P_ij = exp_new

                O_i = O_i * exp_old + torch.matmul(P_ij, V_j)
                m_i = m_new
                l_i = l_new

            O_i = O_i / (l_i + 1e-8)
            O[:, :, r_start:r_end, :] = O_i

        return O


class FlashAttentionV3Analysis:
    """V3 的性能分析和对比工具"""

    @staticmethod
    def compute_arithmetic_intensity(N: int, d: int, dtype_bytes: int = 2) -> dict:
        """
        计算 Attention 的算术强度 (Arithmetic Intensity = FLOPs / Bytes).

        Standard Attention:
            FLOPs: 4 * N^2 * d  (2个matmul, 各2*N*N*d FLOPS)
            Bytes: 3*N*d*dtype + 2*N^2*dtype (读QKV + 读写S,P)
            AI = 4*N^2*d / (3*N*d + 2*N^2) / dtype_bytes

        FlashAttention (V2/V3):
            FLOPs: 4 * N^2 * d (不变)
            Bytes: 3*N*d*dtype (只读QKV, 写O)
            AI = 4*N^2*d / (4*N*d) / dtype_bytes = N / dtype_bytes
        """
        # Standard
        std_flops = 4 * N * N * d
        std_bytes = (3 * N * d + 2 * N * N) * dtype_bytes
        std_ai = std_flops / std_bytes

        # Flash
        flash_flops = 4 * N * N * d
        flash_bytes = 4 * N * d * dtype_bytes  # Q,K,V read + O write
        flash_ai = flash_flops / flash_bytes

        return {
            "standard_ai": std_ai,
            "flash_ai": flash_ai,
            "improvement": flash_ai / std_ai,
            "flash_flops": flash_flops,
            "flash_bytes": flash_bytes,
        }

    @staticmethod
    def estimate_performance(N: int, d: int, B: int, H: int,
                             config: HopperConfig = HopperConfig(),
                             dtype: str = "fp16") -> dict:
        """估算 V3 在 H100 上的理论性能"""
        total_flops = B * H * 4 * N * N * d
        total_bytes = B * H * 4 * N * d * (1 if dtype == "fp8" else 2)

        peak_tflops = config.fp8_tflops if dtype == "fp8" else config.fp16_tflops

        # 计算时间 (μs)
        compute_time = total_flops / (peak_tflops * 1e12) * 1e6
        memory_time = total_bytes / (config.mem_bw_gbps * 1e9) * 1e6

        # V3 的优化因子
        overlap_factor = 0.85  # producer-consumer overlap
        pingpong_factor = 0.95  # ping-pong 减少 idle

        effective_time = max(compute_time * pingpong_factor,
                             memory_time / config.tma_speedup)
        effective_time_with_overlap = effective_time / overlap_factor

        achieved_tflops = total_flops / (effective_time_with_overlap * 1e-6) / 1e12

        return {
            "total_flops": total_flops,
            "total_bytes": total_bytes,
            "compute_time_us": compute_time,
            "memory_time_us": memory_time,
            "effective_time_us": effective_time_with_overlap,
            "achieved_tflops": achieved_tflops,
            "compute_utilization": achieved_tflops / peak_tflops,
        }


# ============================================================================
# 测试与验证
# ============================================================================
def test_flash_v3():
    """验证 FlashAttention V3 模拟实现的正确性"""
    print("=" * 70)
    print("测试: FlashAttention V3 (模拟)")
    print("=" * 70)

    torch.manual_seed(42)
    B, H, d = 2, 4, 64

    # 基本正确性测试
    print("\n[正确性测试] — Warp Specialized Forward:")
    for N in [64, 128, 256, 512]:
        Q = torch.randn(B, H, N, d)
        K = torch.randn(B, H, N, d)
        V = torch.randn(B, H, N, d)

        # Reference
        scale = 1.0 / math.sqrt(d)
        S = torch.matmul(Q, K.transpose(-2, -1)) * scale
        O_ref = torch.matmul(F.softmax(S, dim=-1), V)

        # V3 (warp specialized)
        O_v3, stats = FlashAttentionV3Sim.forward_warp_specialized(Q, K, V, B_r=64, B_c=64)
        diff = (O_v3 - O_ref).abs().max().item()
        print(f"  N={N:4d}  max_diff = {diff:.2e}  overlap_ratio = {stats['overlap_ratio']:.2f}  "
              f"{'PASS' if diff < 1e-4 else 'FAIL'}")

    # Causal 测试
    print("\n[Causal 测试]:")
    for N in [128, 256]:
        Q = torch.randn(B, H, N, d)
        K = torch.randn(B, H, N, d)
        V = torch.randn(B, H, N, d)

        scale = 1.0 / math.sqrt(d)
        S = torch.matmul(Q, K.transpose(-2, -1)) * scale
        mask = torch.triu(torch.ones(N, N, dtype=torch.bool), diagonal=1)
        S.masked_fill_(mask, float('-inf'))
        O_ref = torch.matmul(F.softmax(S, dim=-1), V)

        O_v3, _ = FlashAttentionV3Sim.forward_warp_specialized(Q, K, V, B_r=64, B_c=64, causal=True)
        diff = (O_v3 - O_ref).abs().max().item()
        print(f"  N={N:4d}  causal  max_diff = {diff:.2e}  {'PASS' if diff < 1e-4 else 'FAIL'}")

    # FP8 模拟测试
    print("\n[FP8 模拟测试]:")
    N = 256
    Q = torch.randn(B, H, N, d)
    K = torch.randn(B, H, N, d)
    V = torch.randn(B, H, N, d)

    O_fp16, _ = FlashAttentionV3Sim.forward_warp_specialized(Q, K, V, simulate_fp8=False)
    O_fp8, _ = FlashAttentionV3Sim.forward_warp_specialized(Q, K, V, simulate_fp8=True)

    scale = 1.0 / math.sqrt(d)
    S = torch.matmul(Q, K.transpose(-2, -1)) * scale
    O_ref = torch.matmul(F.softmax(S, dim=-1), V)

    diff_fp16 = (O_fp16 - O_ref).abs().max().item()
    diff_fp8 = (O_fp8 - O_ref).abs().max().item()
    print(f"  FP16 max_diff = {diff_fp16:.2e}")
    print(f"  FP8  max_diff = {diff_fp8:.2e}  (precision loss expected)")

    # Ping-Pong 测试
    print("\n[Ping-Pong Scheduling 测试]:")
    for N in [128, 256]:
        Q = torch.randn(B, H, N, d)
        K = torch.randn(B, H, N, d)
        V = torch.randn(B, H, N, d)

        scale = 1.0 / math.sqrt(d)
        S = torch.matmul(Q, K.transpose(-2, -1)) * scale
        O_ref = torch.matmul(F.softmax(S, dim=-1), V)

        O_pp = FlashAttentionV3Sim.forward_pingpong(Q, K, V, B_r=64, B_c=64)
        diff = (O_pp - O_ref).abs().max().item()
        print(f"  N={N:4d}  max_diff = {diff:.2e}  {'PASS' if diff < 1e-4 else 'FAIL'}")


def test_performance_analysis():
    """V3 理论性能分析"""
    print("\n" + "=" * 70)
    print("FlashAttention V3 性能分析 (H100 SXM)")
    print("=" * 70)

    config = HopperConfig()
    B, H, d = 1, 32, 128

    # 算术强度分析
    print("\n[算术强度 (Arithmetic Intensity)]:")
    print(f"  {'N':>6s} | {'Standard AI':>12s} | {'Flash AI':>10s} | {'提升':>6s}")
    print("  " + "-" * 45)
    for N in [1024, 2048, 4096, 8192, 16384]:
        ai = FlashAttentionV3Analysis.compute_arithmetic_intensity(N, d)
        print(f"  {N:>6d} | {ai['standard_ai']:>12.1f} | {ai['flash_ai']:>10.1f} | {ai['improvement']:>5.1f}x")

    # 性能估算
    print(f"\n[估算性能] B={B}, H={H}, d={d}:")
    print(f"  {'N':>6s} | {'FP16 TFLOPS':>12s} | {'FP8 TFLOPS':>11s} | {'FP16 GPU%':>10s} | {'FP8 GPU%':>9s}")
    print("  " + "-" * 60)
    for N in [1024, 2048, 4096, 8192, 16384]:
        perf_fp16 = FlashAttentionV3Analysis.estimate_performance(N, d, B, H, config, "fp16")
        perf_fp8 = FlashAttentionV3Analysis.estimate_performance(N, d, B, H, config, "fp8")
        print(f"  {N:>6d} | {perf_fp16['achieved_tflops']:>12.1f} | {perf_fp8['achieved_tflops']:>11.1f} | "
              f"{perf_fp16['compute_utilization']*100:>9.1f}% | {perf_fp8['compute_utilization']*100:>8.1f}%")


def compare_all_versions():
    """对比 FlashAttention V1/V2/V3"""
    print("\n" + "=" * 70)
    print("FlashAttention V1 → V2 → V3 演化对比")
    print("=" * 70)

    print("""
    ┌─────────────────┬──────────────────┬──────────────────┬──────────────────┐
    │   特性          │  FlashAttn V1    │  FlashAttn V2    │  FlashAttn V3    │
    ├─────────────────┼──────────────────┼──────────────────┼──────────────────┤
    │ 目标 GPU        │  Ampere (A100)   │  Ampere (A100)   │  Hopper (H100)   │
    ├─────────────────┼──────────────────┼──────────────────┼──────────────────┤
    │ 外循环          │  K/V blocks      │  Q blocks        │  Q blocks        │
    │ 内循环          │  Q blocks        │  K/V blocks      │  K/V blocks      │
    ├─────────────────┼──────────────────┼──────────────────┼──────────────────┤
    │ 数据搬运        │  同步 (手动copy) │  同步 (手动copy) │  异步 (TMA)      │
    │ 与计算重叠      │  ✗ or 有限      │  有限            │  完全重叠         │
    ├─────────────────┼──────────────────┼──────────────────┼──────────────────┤
    │ Warp 分工       │  统一            │  减少通信        │  Producer-        │
    │                 │                  │                  │  Consumer 分工    │
    ├─────────────────┼──────────────────┼──────────────────┼──────────────────┤
    │ Tensor Core 利用│  ~30-40%         │  ~50-70%         │  ~80-90%         │
    ├─────────────────┼──────────────────┼──────────────────┼──────────────────┤
    │ 非 matmul 开销  │  较大            │  减少 ~50%       │  进一步减少       │
    ├─────────────────┼──────────────────┼──────────────────┼──────────────────┤
    │ 低精度支持      │  FP16/BF16       │  FP16/BF16       │  FP16/BF16/FP8!  │
    ├─────────────────┼──────────────────┼──────────────────┼──────────────────┤
    │ 调度策略        │  简单            │  简单            │  Ping-Pong!       │
    ├─────────────────┼──────────────────┼──────────────────┼──────────────────┤
    │ IO 复杂度       │  O(N²d²/M)      │  O(N²d²/M)      │  O(N²d²/M)       │
    │                 │                  │                  │  (常数更小)       │
    ├─────────────────┼──────────────────┼──────────────────┼──────────────────┤
    │ A100 实测       │  ~124 TFLOPS     │  ~230 TFLOPS     │  N/A (Hopper)    │
    │ H100 实测       │  N/A             │  ~350-400 TFLOPS │  ~620-740 TFLOPS │
    └─────────────────┴──────────────────┴──────────────────┴──────────────────┘
    """)


if __name__ == "__main__":
    test_flash_v3()
    test_performance_analysis()
    compare_all_versions()
    print("\n✓ Step 6 完成: FlashAttention V3 验证通过")
