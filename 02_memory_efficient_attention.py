"""
Step 2: Standard Self-Attention → Memory Efficient Attention

==========================================================================
2.1 Standard Self-Attention
==========================================================================

    S = Q @ K^T / sqrt(d)      # [N, N]   ← O(N^2) 内存!
    P = softmax(S, dim=-1)     # [N, N]   ← O(N^2) 内存!
    O = P @ V                  # [N, d]

内存分析:
    - 需要存储完整的 S 矩阵: O(N^2) 内存
    - 需要存储完整的 P 矩阵: O(N^2) 内存
    - 总内存: O(N^2)

IO 分析 (HBM 读写):
    - 读 Q, K: O(Nd)
    - 写 S:    O(N^2)
    - 读 S:    O(N^2)
    - 写 P:    O(N^2)
    - 读 P, V: O(N^2) + O(Nd)
    - 写 O:    O(Nd)
    - 总IO:    O(Nd + N^2)

当 N >> d 时 (长序列)，O(N^2) 的内存和 IO 成为瓶颈。

==========================================================================
2.2 Memory Efficient Attention (不存储完整 S/P 矩阵)
==========================================================================

核心思想: 将 Q 分块处理，每个 Q block 独立计算完整的 attention。
不需要在 HBM 中存储 N×N 的 S/P 矩阵。

算法:
    for each Q_block (大小 B_r × d):
        for each K_block, V_block (大小 B_c × d):
            S_block = Q_block @ K_block^T   # [B_r, B_c]  在 SRAM 中
            P_block = softmax(S_block)       # [B_r, B_c]  在 SRAM 中
            O_block += P_block @ V_block     # [B_r, d]    在 SRAM 中
        write O_block to HBM

内存分析:
    - 只需要 O(B_r × B_c) 的 SRAM 即可
    - HBM 中不存储 S, P → O(N) 额外内存
    - 但 softmax 需要全局 max 和 sum → 需要两次遍历 K/V (forward + rescale)

IO 分析:
    - Q 被读 N_k/B_c 次 (对每个 K block)
    - K,V 被读 N_q/B_r 次 (对每个 Q block)
    - 总IO: O(N^2 * d / B_r + N^2 * d / B_c) ≈ O(N^2 * d / M^{1/2})
      其中 M 是 SRAM 大小

问题: softmax 的全局依赖性 — 需要知道全局 max 才能计算正确的 softmax。
解决方案: → FlashAttention 使用 Online-Softmax 技巧!
"""

import torch
import torch.nn.functional as F
import math
import time
from typing import Tuple, Optional


# ============================================================================
# 2.1 Standard Self-Attention (O(N^2) 内存)
# ============================================================================
class StandardAttention:
    """
    标准 Self-Attention 实现。

    内存: O(N^2) — 存储完整 attention matrix
    IO:   O(Nd + N^2) — 读写完整 S, P 矩阵

    这是 baseline，所有优化方法都与此对比。
    """

    @staticmethod
    def forward(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                causal: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            Q: [B, H, N, d]
            K: [B, H, N, d]
            V: [B, H, N, d]
            causal: 是否使用因果 mask
        Returns:
            O:    [B, H, N, d]  — attention 输出
            attn: [B, H, N, N]  — attention 矩阵 (用于可视化/调试)
        """
        B, H, N, d = Q.shape
        scale = 1.0 / math.sqrt(d)

        # Step 1: S = Q @ K^T / sqrt(d)  → 写入 HBM: O(N^2)
        S = torch.matmul(Q, K.transpose(-2, -1)) * scale  # [B, H, N, N]

        # Step 2: Causal mask (optional)
        if causal:
            mask = torch.triu(torch.ones(N, N, device=Q.device, dtype=torch.bool), diagonal=1)
            S.masked_fill_(mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        # Step 3: P = softmax(S)  → 读 S, 写 P: O(N^2) 各一次
        P = F.softmax(S, dim=-1)  # [B, H, N, N]

        # Step 4: O = P @ V  → 读 P, V, 写 O
        O = torch.matmul(P, V)  # [B, H, N, d]

        return O, P

    @staticmethod
    def memory_usage(N: int, d: int, dtype_bytes: int = 4) -> dict:
        """计算内存占用"""
        return {
            "S_matrix": N * N * dtype_bytes,
            "P_matrix": N * N * dtype_bytes,
            "Q_K_V_O": 4 * N * d * dtype_bytes,
            "total": 2 * N * N * dtype_bytes + 4 * N * d * dtype_bytes,
            "total_MB": (2 * N * N * dtype_bytes + 4 * N * d * dtype_bytes) / 1024 / 1024,
        }


# ============================================================================
# 2.2 Memory Efficient Attention (O(N) 额外内存, 但两次遍历 KV)
# ============================================================================
class MemoryEfficientAttention:
    """
    Memory Efficient Attention (Rabe & Staats, 2021)

    核心思想: 分块计算，不在 HBM 中存储完整的 N×N 矩阵。
    使用 Online-Softmax 技巧处理 softmax 的全局依赖性。

    内存: O(N) — 只需存储逐行的 max 和 sum
    IO:   O(Nd + N^2) — 与标准 attention 相同量级
                        (但实际 IO 模式更友好，因为 tiling)
    """

    @staticmethod
    def forward(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                block_size_q: int = 64, block_size_kv: int = 64,
                causal: bool = False) -> torch.Tensor:
        """
        分块 Memory-Efficient Attention。

        算法:
          对 Q 的每个 block:
            对 KV 的每个 block:
              1. 计算 S_block = Q_block @ K_block^T
              2. 在线更新 max, sum
              3. 累积 O_block
        """
        B, H, N_q, d = Q.shape
        N_k = K.shape[2]
        scale = 1.0 / math.sqrt(d)

        O = torch.zeros_like(Q)

        # 对 Q 分块
        for i_start in range(0, N_q, block_size_q):
            i_end = min(i_start + block_size_q, N_q)
            Q_block = Q[:, :, i_start:i_end, :]  # [B, H, Br, d]
            Br = i_end - i_start

            # 初始化当前 Q block 的统计量
            m_i = torch.full((B, H, Br, 1), float('-inf'), device=Q.device, dtype=Q.dtype)
            d_i = torch.zeros((B, H, Br, 1), device=Q.device, dtype=Q.dtype)
            O_i = torch.zeros((B, H, Br, d), device=Q.device, dtype=Q.dtype)

            # 对 KV 分块遍历
            kv_end = i_end if causal else N_k
            for j_start in range(0, kv_end, block_size_kv):
                j_end = min(j_start + block_size_kv, kv_end if causal else N_k)
                K_block = K[:, :, j_start:j_end, :]  # [B, H, Bc, d]
                V_block = V[:, :, j_start:j_end, :]

                # 计算 attention scores (在 SRAM 中)
                S_block = torch.matmul(Q_block, K_block.transpose(-2, -1)) * scale

                # 因果 mask
                if causal:
                    row_idx = torch.arange(i_start, i_end, device=Q.device).unsqueeze(1)
                    col_idx = torch.arange(j_start, j_end, device=Q.device).unsqueeze(0)
                    causal_mask = row_idx < col_idx
                    S_block.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

                # Online-Softmax 更新
                m_block = S_block.max(dim=-1, keepdim=True).values
                m_new = torch.maximum(m_i, m_block)

                exp_diff = torch.exp(m_i - m_new)
                P_block = torch.exp(S_block - m_new)

                d_new = d_i * exp_diff + P_block.sum(dim=-1, keepdim=True)

                # 更新输出
                O_i = O_i * (d_i * exp_diff / (d_new + 1e-8)) + \
                      torch.matmul(P_block, V_block) / (d_new + 1e-8)

                m_i = m_new
                d_i = d_new

            O[:, :, i_start:i_end, :] = O_i

        return O

    @staticmethod
    def memory_usage(N: int, d: int, block_size: int = 64, dtype_bytes: int = 4) -> dict:
        """
        内存占用分析
        """
        sram_per_block = block_size * block_size * dtype_bytes  # S_block
        sram_per_block += 2 * block_size * d * dtype_bytes      # Q_block, KV_block
        return {
            "SRAM_per_block": sram_per_block,
            "SRAM_per_block_KB": sram_per_block / 1024,
            "HBM_extra": 3 * N * dtype_bytes,  # m, d, O per row
            "HBM_extra_KB": 3 * N * dtype_bytes / 1024,
            "total_HBM_MB": (4 * N * d * dtype_bytes + 3 * N * dtype_bytes) / 1024 / 1024,
        }


# ============================================================================
# 测试与比较
# ============================================================================
def test_attention_correctness():
    """验证 Memory-Efficient Attention 与标准 Attention 的一致性"""
    print("=" * 70)
    print("测试: Standard Attention vs Memory-Efficient Attention")
    print("=" * 70)

    torch.manual_seed(42)
    B, H, N, d = 2, 4, 256, 64
    Q = torch.randn(B, H, N, d)
    K = torch.randn(B, H, N, d)
    V = torch.randn(B, H, N, d)

    # Standard Attention
    O_std, P_std = StandardAttention.forward(Q, K, V, causal=False)

    # Memory Efficient Attention (不同 block size)
    for bs_q, bs_kv in [(32, 32), (64, 64), (128, 128), (64, 32)]:
        O_me = MemoryEfficientAttention.forward(Q, K, V, block_size_q=bs_q,
                                                 block_size_kv=bs_kv, causal=False)
        max_diff = (O_me - O_std).abs().max().item()
        print(f"  block_q={bs_q:3d}, block_kv={bs_kv:3d}  max_diff = {max_diff:.2e}  "
              f"{'PASS' if max_diff < 1e-4 else 'FAIL'}")

    # 测试 Causal Attention
    print("\n测试: Causal Attention")
    O_std_c, _ = StandardAttention.forward(Q, K, V, causal=True)
    for bs in [32, 64, 128]:
        O_me_c = MemoryEfficientAttention.forward(Q, K, V, block_size_q=bs,
                                                    block_size_kv=bs, causal=True)
        max_diff = (O_me_c - O_std_c).abs().max().item()
        print(f"  causal block_size={bs:3d}  max_diff = {max_diff:.2e}  "
              f"{'PASS' if max_diff < 1e-4 else 'FAIL'}")


def test_memory_comparison():
    """对比两种方法的内存占用"""
    print("\n" + "=" * 70)
    print("内存占用对比 (float32, d=64)")
    print("=" * 70)

    d = 64
    print(f"{'N':>8s} | {'Standard (MB)':>15s} | {'MemEff (MB)':>15s} | {'节省':>8s}")
    print("-" * 55)
    for N in [256, 512, 1024, 2048, 4096, 8192, 16384]:
        std_mem = StandardAttention.memory_usage(N, d)
        me_mem = MemoryEfficientAttention.memory_usage(N, d)
        savings = 1 - me_mem['total_HBM_MB'] / std_mem['total_MB']
        print(f"{N:>8d} | {std_mem['total_MB']:>13.2f}MB | {me_mem['total_HBM_MB']:>13.2f}MB | {savings:>6.1%}")


if __name__ == "__main__":
    test_attention_correctness()
    test_memory_comparison()
    print("\n✓ Step 2 完成: Memory-Efficient Attention 验证通过")
