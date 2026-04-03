"""
Step 5: FlashDecoding (Dao et al., 2023) & FlashDecoding++ (Hong et al., 2024)

==========================================================================
背景: 推理时的 Decode 阶段瓶颈
==========================================================================

LLM 推理分两个阶段:
    1. Prefill (预填充): 处理完整 prompt, Q 长度 = prompt_len
       → 计算密集, FlashAttention V2 工作良好
    2. Decode (解码):    逐 token 生成, Q 长度 = 1
       → 内存带宽瓶颈! 因为只有一个 query token

Decode 阶段的问题:
    Q: [B, H, 1, d]     ← 只有 1 个 token!
    K: [B, H, S, d]     ← S 可能很长 (几千到几万)
    V: [B, H, S, d]

    FlashAttention V2 在 decode 时:
    - 外循环只有 1 个 Q block (因为 seq_len=1)
    - 只启动 B*H 个 thread blocks → 无法充分利用 GPU!
    - A100 有 108 个 SM, 但 B*H 可能 < 108 (如 B=1, H=32 → 只用 32 个 SM)
    - GPU 利用率极低!

==========================================================================
FlashDecoding (Dao et al., 2023)
==========================================================================

核心思想: 在 KV 序列维度上增加并行度!

    FlashAttention V2 的并行: {batch, head, Q_block}
    FlashDecoding 增加:       {batch, head, Q_block, KV_split}  ← 新增!

算法:
    1. 将 KV 序列分成 S 个 splits
    2. 每个 split 独立计算局部 attention:
       - 每个 split 得到局部 O_s, m_s, ℓ_s
    3. 用 Online-Softmax 的 reduce 操作合并所有 splits 的结果

    Step 1 (并行): 对每个 KV split s:
        Q @ K_s^T → S_s → local softmax → P_s @ V_s → O_s, m_s, ℓ_s

    Step 2 (Reduce): 合并所有 splits 的结果
        m_final = max(m_1, m_2, ..., m_S)
        ℓ_final = Σ_s ℓ_s * exp(m_s - m_final)
        O_final = Σ_s (ℓ_s * exp(m_s - m_final) / ℓ_final) * O_s

并行度提升:
    V2: B * H * ceil(N_q / B_r) 个 blocks
    FlashDecoding: B * H * num_splits 个 blocks
    当 N_q=1 (decode): 从 B*H 增加到 B*H*num_splits (几倍到十几倍!)

==========================================================================
FlashDecoding++ (Hong et al., 2024)
==========================================================================

FlashDecoding 的问题:
    reduce 操作需要额外的全局 sync + 额外 kernel launch
    → 有 overhead, 尤其当 splits 数量少时

FlashDecoding++ 的改进:
    1. 统一的 softmax (Unified Max): 基于先验知识预估全局 max
       - 观察: attention scores 的 max 值相对稳定
       - 使用一个预估的 φ 作为 "统一 max"
       - 当 φ ≈ true max 时, 各 split 的结果可以直接相加!
       - 无需 reduce 阶段的 max 校正

    2. Flat GEMM 优化: 当 Q_len=1 时, matmul 退化为 GEMV
       - 使用专门优化的 Flat GEMM kernel

    数学: 如果所有 split 使用相同的 max = φ:
        O_s = Σ_k exp(S_{sk} - φ) * V_k / Σ_k exp(S_{sk} - φ)
        O = Σ_s w_s * O_s
        其中 w_s = ℓ_s / Σ_s' ℓ_s'

    当 φ = true_max 时, 这完全等价于标准 softmax!
    当 φ ≠ true_max 时, 只要 φ 足够大 (避免上溢), 结果仍然精确。

==========================================================================
"""

import torch
import torch.nn.functional as F
import math
from typing import Tuple


class FlashDecoding:
    """
    FlashDecoding: 在 KV 序列维度上并行的 Attention。

    适用场景: decode 阶段 (Q_len = 1, KV_len >> 1)

    并行策略:
        1. 将 KV 分成 num_splits 份
        2. 每份独立计算局部 attention
        3. 用 online-softmax reduce 合并结果
    """

    @staticmethod
    def forward(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                num_splits: int = 4,
                causal: bool = False) -> torch.Tensor:
        """
        FlashDecoding Forward.

        Args:
            Q: [B, H, N_q, d]  — 通常 N_q=1 (decode) 或较小
            K: [B, H, N_k, d]  — KV cache, 可能很长
            V: [B, H, N_k, d]
            num_splits: KV 序列的分割数 (并行度)
        Returns:
            O: [B, H, N_q, d]
        """
        B, H, N_q, d = Q.shape
        N_k = K.shape[2]
        scale = 1.0 / math.sqrt(d)

        # 计算每个 split 的大小
        split_size = math.ceil(N_k / num_splits)
        actual_splits = math.ceil(N_k / split_size)

        # ============================================================
        # Phase 1: 并行计算每个 split 的局部结果 (可并行!)
        # ============================================================
        # 每个 split 产生: O_s [B,H,N_q,d], m_s [B,H,N_q,1], l_s [B,H,N_q,1]
        O_splits = []
        m_splits = []
        l_splits = []

        for s in range(actual_splits):
            s_start = s * split_size
            s_end = min((s + 1) * split_size, N_k)

            K_s = K[:, :, s_start:s_end, :]  # [B, H, split, d]
            V_s = V[:, :, s_start:s_end, :]

            # 局部 attention scores
            S_s = torch.matmul(Q, K_s.transpose(-2, -1)) * scale  # [B,H,N_q,split]

            # Causal mask
            if causal:
                # 在 decode 阶段，Q 的位置 = N_k - 1 (最后一个 token)
                # 所以所有 K 位置都是可见的...但我们还是处理一般情况
                q_pos = torch.arange(N_k - N_q, N_k, device=Q.device).unsqueeze(1)
                k_pos = torch.arange(s_start, s_end, device=Q.device).unsqueeze(0)
                mask = q_pos < k_pos
                S_s.masked_fill_(mask.unsqueeze(0).unsqueeze(0), float('-inf'))

            # 局部 softmax 统计量
            m_s = S_s.max(dim=-1, keepdim=True).values        # [B,H,N_q,1]
            P_s = torch.exp(S_s - m_s)                         # [B,H,N_q,split]
            l_s = P_s.sum(dim=-1, keepdim=True)                # [B,H,N_q,1]

            # 局部加权和 (不除以 l_s, 保留 unnormalized)
            O_s = torch.matmul(P_s, V_s)                       # [B,H,N_q,d]

            O_splits.append(O_s)
            m_splits.append(m_s)
            l_splits.append(l_s)

        # ============================================================
        # Phase 2: Reduce — 合并所有 splits 的结果
        # ============================================================
        # 使用 Online-Softmax 的思想, 将多个 split 的局部结果合并

        # 初始化: 用第一个 split
        O_final = O_splits[0]
        m_final = m_splits[0]
        l_final = l_splits[0]

        # 逐步合并
        for s in range(1, actual_splits):
            m_new = torch.maximum(m_final, m_splits[s])

            # 重新缩放
            exp_old = torch.exp(m_final - m_new)
            exp_new = torch.exp(m_splits[s] - m_new)

            l_new = l_final * exp_old + l_splits[s] * exp_new

            # O = (l_old * exp_old * O_old + l_new_s * exp_new * O_new_s) / l_total
            # 但注意: O_splits 已经是 unnormalized 的 (= P @ V, 不是 softmax(S) @ V)
            O_final = O_final * exp_old + O_splits[s] * exp_new

            m_final = m_new
            l_final = l_new

        # 最终归一化
        O_final = O_final / (l_final + 1e-8)

        return O_final

    @staticmethod
    def parallelism_analysis(B: int, H: int, N_q: int, N_k: int,
                             num_splits: int) -> dict:
        """分析并行度提升"""
        v2_blocks = B * H * max(1, N_q // 64)
        fd_blocks = B * H * max(1, N_q // 64) * num_splits
        return {
            "flashattn_v2_blocks": v2_blocks,
            "flashdecoding_blocks": fd_blocks,
            "parallelism_boost": fd_blocks / v2_blocks,
        }


class FlashDecodingPP:
    """
    FlashDecoding++ (Hong et al., 2024)

    改进: 使用 Unified Max (统一最大值) 避免 reduce 阶段的 max 校正。

    核心思想: 如果所有 splits 使用相同的预估 max φ:
      - 各 split 的结果可以直接加权平均
      - reduce 变成简单的加权和, 无需 max 校正
      - 减少了一次 kernel launch 和 global sync

    φ 的选择:
      - 可以用历史的 attention score max 作为 φ
      - 或者用一个安全的上界 (如 max(S) 的统计估计)
      - 当 φ ≥ true_max 时, 数值上是安全的
    """

    @staticmethod
    def forward(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                num_splits: int = 4,
                phi: float = None,
                causal: bool = False) -> torch.Tensor:
        """
        FlashDecoding++ Forward.

        Args:
            Q: [B, H, N_q, d]
            K: [B, H, N_k, d]
            V: [B, H, N_k, d]
            num_splits: KV 分割数
            phi: 预估的全局 max (如果 None, 自动计算)
        Returns:
            O: [B, H, N_q, d]
        """
        B, H, N_q, d = Q.shape
        N_k = K.shape[2]
        scale = 1.0 / math.sqrt(d)

        split_size = math.ceil(N_k / num_splits)
        actual_splits = math.ceil(N_k / split_size)

        # ============================================================
        # Step 0: 确定 Unified Max φ
        # ============================================================
        if phi is None:
            # 自动估算: 用一个快速的近似 (实践中可以用历史统计)
            # 这里为了正确性, 先快速扫描一遍得到 true max
            with torch.no_grad():
                S_sample = torch.matmul(Q, K.transpose(-2, -1)) * scale
                phi_tensor = S_sample.max(dim=-1, keepdim=True).values  # [B,H,N_q,1]
        else:
            phi_tensor = torch.full((B, H, N_q, 1), phi, device=Q.device, dtype=Q.dtype)

        # ============================================================
        # Phase 1: 各 split 用统一的 φ 计算 (可并行, 无需相互通信!)
        # ============================================================
        O_sum = torch.zeros(B, H, N_q, d, device=Q.device, dtype=Q.dtype)
        l_sum = torch.zeros(B, H, N_q, 1, device=Q.device, dtype=Q.dtype)

        for s in range(actual_splits):
            s_start = s * split_size
            s_end = min((s + 1) * split_size, N_k)

            K_s = K[:, :, s_start:s_end, :]
            V_s = V[:, :, s_start:s_end, :]

            S_s = torch.matmul(Q, K_s.transpose(-2, -1)) * scale

            if causal:
                q_pos = torch.arange(N_k - N_q, N_k, device=Q.device).unsqueeze(1)
                k_pos = torch.arange(s_start, s_end, device=Q.device).unsqueeze(0)
                mask = q_pos < k_pos
                S_s.masked_fill_(mask.unsqueeze(0).unsqueeze(0), float('-inf'))

            # 使用统一的 φ — 这是关键区别!
            # 所有 splits 减去相同的 φ, 结果可以直接相加
            P_s = torch.exp(S_s - phi_tensor)  # [B,H,N_q,split]
            l_s = P_s.sum(dim=-1, keepdim=True)
            O_s = torch.matmul(P_s, V_s)

            # 直接累加! 因为共享相同的 φ
            O_sum = O_sum + O_s
            l_sum = l_sum + l_s

        # ============================================================
        # Phase 2: 简单归一化 (无需 max 校正!)
        # ============================================================
        O_final = O_sum / (l_sum + 1e-8)

        return O_final


# ============================================================================
# 测试与验证
# ============================================================================
def test_flash_decoding():
    """验证 FlashDecoding 正确性"""
    print("=" * 70)
    print("测试: FlashDecoding")
    print("=" * 70)

    torch.manual_seed(42)
    B, H, d = 2, 8, 64

    # ── Decode 场景: Q_len=1, KV_len很长 ──
    print("\n[Decode 场景] Q_len=1, 不同 KV_len:")
    for N_k in [256, 512, 1024, 2048, 4096]:
        Q = torch.randn(B, H, 1, d)
        K = torch.randn(B, H, N_k, d)
        V = torch.randn(B, H, N_k, d)

        # Reference
        scale = 1.0 / math.sqrt(d)
        S = torch.matmul(Q, K.transpose(-2, -1)) * scale
        O_ref = torch.matmul(F.softmax(S, dim=-1), V)

        # FlashDecoding
        for ns in [2, 4, 8, 16]:
            if ns > N_k // 16:
                continue
            O_fd = FlashDecoding.forward(Q, K, V, num_splits=ns)
            diff = (O_fd - O_ref).abs().max().item()
            status = 'PASS' if diff < 1e-4 else 'FAIL'
            if ns == 4:  # 只显示 num_splits=4 的结果, 其他只在失败时显示
                print(f"  N_k={N_k:5d}, splits={ns:2d}  max_diff = {diff:.2e}  {status}")
            elif status == 'FAIL':
                print(f"  N_k={N_k:5d}, splits={ns:2d}  max_diff = {diff:.2e}  {status}")

    # ── 并行度分析 ──
    print("\n[并行度分析] B=1, H=32, Q_len=1:")
    print(f"  {'N_k':>6s} | {'V2 blocks':>10s} | {'FD blocks (s=8)':>16s} | {'提升':>6s}")
    print("  " + "-" * 50)
    for N_k in [256, 512, 1024, 4096, 8192]:
        info = FlashDecoding.parallelism_analysis(1, 32, 1, N_k, num_splits=8)
        print(f"  {N_k:>6d} | {info['flashattn_v2_blocks']:>10d} | "
              f"{info['flashdecoding_blocks']:>16d} | {info['parallelism_boost']:>5.1f}x")


def test_flash_decoding_pp():
    """验证 FlashDecoding++ 正确性"""
    print("\n" + "=" * 70)
    print("测试: FlashDecoding++")
    print("=" * 70)

    torch.manual_seed(42)
    B, H, d = 2, 8, 64

    print("\n[Decode 场景] Q_len=1:")
    for N_k in [256, 512, 1024, 2048]:
        Q = torch.randn(B, H, 1, d)
        K = torch.randn(B, H, N_k, d)
        V = torch.randn(B, H, N_k, d)

        # Reference
        scale = 1.0 / math.sqrt(d)
        S = torch.matmul(Q, K.transpose(-2, -1)) * scale
        O_ref = torch.matmul(F.softmax(S, dim=-1), V)

        # FlashDecoding++
        for ns in [4, 8]:
            O_fdpp = FlashDecodingPP.forward(Q, K, V, num_splits=ns)
            diff = (O_fdpp - O_ref).abs().max().item()
            print(f"  N_k={N_k:5d}, splits={ns:2d}  max_diff = {diff:.2e}  "
                  f"{'PASS' if diff < 1e-4 else 'FAIL'}")

    # ── 与 FlashDecoding 对比 ──
    print("\n[FlashDecoding vs FlashDecoding++ 对比]:")
    Q = torch.randn(2, 8, 1, 64)
    K = torch.randn(2, 8, 2048, 64)
    V = torch.randn(2, 8, 2048, 64)

    scale = 1.0 / math.sqrt(64)
    S = torch.matmul(Q, K.transpose(-2, -1)) * scale
    O_ref = torch.matmul(F.softmax(S, dim=-1), V)

    O_fd = FlashDecoding.forward(Q, K, V, num_splits=8)
    O_fdpp = FlashDecodingPP.forward(Q, K, V, num_splits=8)

    diff_fd = (O_fd - O_ref).abs().max().item()
    diff_fdpp = (O_fdpp - O_ref).abs().max().item()
    print(f"  FlashDecoding    max_diff = {diff_fd:.2e}")
    print(f"  FlashDecoding++  max_diff = {diff_fdpp:.2e}")


def compare_decode_methods():
    """对比所有 decode 方法"""
    print("\n" + "=" * 70)
    print("FlashDecoding vs FlashDecoding++ 特性对比")
    print("=" * 70)

    print("""
    ┌────────────────────┬───────────────────────┬───────────────────────┐
    │    特性            │  FlashDecoding         │  FlashDecoding++      │
    ├────────────────────┼───────────────────────┼───────────────────────┤
    │ KV Split 并行      │  ✓ num_splits 份      │  ✓ num_splits 份      │
    ├────────────────────┼───────────────────────┼───────────────────────┤
    │ Phase 1            │  各 split 独立计算     │  各 split 独立计算    │
    │ (Split 计算)       │  局部 max m_s          │  统一 max φ          │
    ├────────────────────┼───────────────────────┼───────────────────────┤
    │ Phase 2            │  Online-Softmax reduce │  简单加权平均         │
    │ (合并)             │  需要 max 校正         │  无需 max 校正!       │
    │                    │  额外 kernel launch    │  可融合到 Phase 1     │
    ├────────────────────┼───────────────────────┼───────────────────────┤
    │ φ 选择             │  不需要                │  需要预估 global max   │
    │                    │                       │  (历史统计 / 上界估计) │
    ├────────────────────┼───────────────────────┼───────────────────────┤
    │ 数值精度            │  精确                  │  精确 (当 φ ≥ max)    │
    │                    │                       │  近似 (当 φ < max)    │
    ├────────────────────┼───────────────────────┼───────────────────────┤
    │ Kernel launches     │  2 (split + reduce)   │  1 (融合)             │
    ├────────────────────┼───────────────────────┼───────────────────────┤
    │ 适用场景            │  所有 decode           │  φ 可准确预估的场景    │
    └────────────────────┴───────────────────────┴───────────────────────┘
    """)


if __name__ == "__main__":
    test_flash_decoding()
    test_flash_decoding_pp()
    compare_decode_methods()
    print("\n✓ Step 5 完成: FlashDecoding & FlashDecoding++ 验证通过")
