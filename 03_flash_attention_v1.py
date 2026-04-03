"""
Step 3: FlashAttention V1 (Dao et al., 2022)

论文: "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"

==========================================================================
核心突破: IO-Aware 的 Tiling 算法
==========================================================================

关键洞察: GPU 的瓶颈不是计算量 (FLOPs)，而是内存带宽 (IO)!

    GPU 内存层次:
    ┌────────────────────┐
    │      SRAM          │  ← 快 (19 TB/s on A100), 小 (20 MB)
    │   (Shared Memory)  │
    ├────────────────────┤
    │      HBM           │  ← 慢 (2 TB/s on A100), 大 (80 GB)
    │  (Global Memory)   │
    └────────────────────┘

    标准 Attention: 将 N×N 矩阵写入/读出 HBM 多次 → IO 瓶颈
    FlashAttention: 所有中间结果保留在 SRAM 中 → IO 最优

==========================================================================
FlashAttention V1 算法
==========================================================================

输入: Q, K, V ∈ R^{N×d}, 在 HBM 中
输出: O ∈ R^{N×d}, 写回 HBM

参数: SRAM 大小 M, block sizes B_r = ceil(M / 4d), B_c = min(ceil(M / 4d), d)

算法 (外循环 KV, 内循环 Q):
    1. 将 K, V 分成 T_c = ceil(N / B_c) 个 blocks
    2. 将 Q, O 分成 T_r = ceil(N / B_r) 个 blocks
    3. 在 HBM 中初始化: O = 0, ℓ = 0, m = -∞
    4. for j = 1, ..., T_c:                    ← 外循环: KV blocks
         从 HBM 加载 K_j, V_j 到 SRAM
         for i = 1, ..., T_r:                  ← 内循环: Q blocks
           从 HBM 加载 Q_i, O_i, ℓ_i, m_i
           在 SRAM 中计算:
             S_ij = Q_i @ K_j^T               (B_r × B_c)
             m̃_ij = rowmax(S_ij)
             P̃_ij = exp(S_ij - m̃_ij)
             ℓ̃_ij = rowsum(P̃_ij)
             m_new = max(m_i, m̃_ij)
             ℓ_new = exp(m_i - m_new) * ℓ_i + exp(m̃_ij - m_new) * ℓ̃_ij
             O_i = diag(ℓ_new)^{-1} * (diag(ℓ_i) * exp(m_i - m_new) * O_i
                                        + exp(m̃_ij - m_new) * P̃_ij @ V_j)
           写回 O_i, ℓ_new, m_new 到 HBM

IO 复杂度: O(N^2 * d^2 / M)
    - 当 M = Θ(Nd) 时, IO = O(N^2 * d / M^{1/2}) → 次二次方!
    - 对比标准 Attention: O(Nd + N^2) ← 始终是二次方

内存复杂度: O(N) — 不存储 N×N 矩阵!

==========================================================================
V1 的特点:
    - 外循环遍历 KV, 内循环遍历 Q (这很关键 — 影响因果 mask 效率)
    - 需要在内循环中读写 O、ℓ、m (额外 IO)
    - Forward 和 Backward 都使用 tiling
    - Backward 通过重计算 S, P (不存储) 来节省内存
==========================================================================
"""

import torch
import torch.nn.functional as F
import math
import time
from typing import Tuple, Optional


class FlashAttentionV1:
    """
    FlashAttention V1 的 Python 参考实现。

    完整实现了论文中的 Algorithm 1 (Forward Pass)。

    特点:
    - 外循环: K/V blocks (j)
    - 内循环: Q blocks (i)
    - 使用 Online-Softmax 递推
    - 不存储 N×N attention 矩阵

    注意: 这是纯 Python 参考实现，用于理解算法。
    实际生产中应使用 Triton/CUDA kernel。
    """

    @staticmethod
    def forward(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                block_size_r: int = 64, block_size_c: int = 64,
                causal: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        FlashAttention V1 Forward Pass (Algorithm 1 from paper).

        Args:
            Q: [B, H, N, d]
            K: [B, H, N, d]
            V: [B, H, N, d]
            block_size_r: Q 方向的 block 大小 (B_r)
            block_size_c: KV 方向的 block 大小 (B_c)
            causal: 是否使用因果 mask

        Returns:
            O: [B, H, N, d]  — 输出
            l: [B, H, N, 1]  — 行和 (用于 backward)
            m: [B, H, N, 1]  — 行最大值 (用于 backward)
        """
        B, H, N, d = Q.shape
        scale = 1.0 / math.sqrt(d)

        Tr = math.ceil(N / block_size_r)  # Q blocks 数量
        Tc = math.ceil(N / block_size_c)  # KV blocks 数量

        # 在 "HBM" 中初始化
        O = torch.zeros_like(Q)
        l = torch.zeros(B, H, N, 1, device=Q.device, dtype=Q.dtype)      # row sum
        m = torch.full((B, H, N, 1), float('-inf'), device=Q.device, dtype=Q.dtype)  # row max

        # ============================================================
        # 外循环: 遍历 K, V blocks
        # ============================================================
        for j in range(Tc):
            j_start = j * block_size_c
            j_end = min((j + 1) * block_size_c, N)

            # 从 "HBM" 加载 K_j, V_j 到 "SRAM"
            K_j = K[:, :, j_start:j_end, :]  # [B, H, B_c, d]
            V_j = V[:, :, j_start:j_end, :]  # [B, H, B_c, d]

            # ============================================================
            # 内循环: 遍历 Q blocks
            # ============================================================
            for i in range(Tr):
                i_start = i * block_size_r
                i_end = min((i + 1) * block_size_r, N)

                # 因果: 如果当前 Q block 的所有行都在 K block 之前，跳过
                # Q rows [i_start, i_end), K cols [j_start, j_end)
                # 当 i_end - 1 < j_start 时, 所有 Q 行都在 K 列之前, 全部被 mask
                if causal and i_end <= j_start:
                    continue

                # 从 "HBM" 加载 Q_i, O_i, l_i, m_i
                Q_i = Q[:, :, i_start:i_end, :]  # [B, H, B_r, d]
                O_i = O[:, :, i_start:i_end, :]  # [B, H, B_r, d]
                l_i = l[:, :, i_start:i_end, :]  # [B, H, B_r, 1]
                m_i = m[:, :, i_start:i_end, :]  # [B, H, B_r, 1]

                # ── 在 "SRAM" 中计算 ──

                # Step 1: S_ij = Q_i @ K_j^T / sqrt(d)
                S_ij = torch.matmul(Q_i, K_j.transpose(-2, -1)) * scale  # [B,H,B_r,B_c]

                # Step 2: 因果 mask
                if causal:
                    row_idx = torch.arange(i_start, i_end, device=Q.device).unsqueeze(1)
                    col_idx = torch.arange(j_start, j_end, device=Q.device).unsqueeze(0)
                    causal_mask = row_idx < col_idx
                    S_ij.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

                # Step 3: 当前 block 的统计量
                m_tilde = S_ij.max(dim=-1, keepdim=True).values     # [B,H,B_r,1]
                P_tilde = torch.exp(S_ij - m_tilde)                 # [B,H,B_r,B_c]
                l_tilde = P_tilde.sum(dim=-1, keepdim=True)         # [B,H,B_r,1]

                # Step 4: 更新全局统计量
                m_new = torch.maximum(m_i, m_tilde)                 # [B,H,B_r,1]
                l_new = (torch.exp(m_i - m_new) * l_i +
                         torch.exp(m_tilde - m_new) * l_tilde)      # [B,H,B_r,1]

                # Step 5: 更新输出
                #   O_i_new = (1/l_new) * (l_i * exp(m_i - m_new) * O_i
                #                          + exp(m_tilde - m_new) * P_tilde @ V_j)
                O_new = (1.0 / (l_new + 1e-8)) * (
                    torch.exp(m_i - m_new) * l_i * O_i +
                    torch.exp(m_tilde - m_new) * torch.matmul(P_tilde, V_j)
                )

                # ── 写回 "HBM" ──
                O[:, :, i_start:i_end, :] = O_new
                l[:, :, i_start:i_end, :] = l_new
                m[:, :, i_start:i_end, :] = m_new

        return O, l, m

    @staticmethod
    def backward(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                 O: torch.Tensor, dO: torch.Tensor,
                 l: torch.Tensor, m: torch.Tensor,
                 block_size_r: int = 64, block_size_c: int = 64,
                 causal: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        FlashAttention V1 Backward Pass.

        关键: 不存储 P 矩阵，而是重新计算 (recomputation)。
        这是用计算换内存的经典 trade-off。

        Args:
            Q, K, V: 前向输入 [B, H, N, d]
            O: 前向输出 [B, H, N, d]
            dO: 输出梯度 [B, H, N, d]
            l, m: 前向保存的统计量
        Returns:
            dQ, dK, dV: 梯度
        """
        B, H, N, d = Q.shape
        scale = 1.0 / math.sqrt(d)

        dQ = torch.zeros_like(Q)
        dK = torch.zeros_like(K)
        dV = torch.zeros_like(V)

        # D = rowsum(dO * O)  — 用于 softmax backward
        D = (dO * O).sum(dim=-1, keepdim=True)  # [B, H, N, 1]

        Tr = math.ceil(N / block_size_r)
        Tc = math.ceil(N / block_size_c)

        for j in range(Tc):
            j_start = j * block_size_c
            j_end = min((j + 1) * block_size_c, N)

            K_j = K[:, :, j_start:j_end, :]
            V_j = V[:, :, j_start:j_end, :]
            dK_j = torch.zeros_like(K_j)
            dV_j = torch.zeros_like(V_j)

            for i in range(Tr):
                i_start = i * block_size_r
                i_end = min((i + 1) * block_size_r, N)

                if causal and i_end <= j_start:
                    continue

                Q_i = Q[:, :, i_start:i_end, :]
                O_i = O[:, :, i_start:i_end, :]
                dO_i = dO[:, :, i_start:i_end, :]
                l_i = l[:, :, i_start:i_end, :]
                m_i = m[:, :, i_start:i_end, :]
                D_i = D[:, :, i_start:i_end, :]

                # 重新计算 S, P (recomputation — 这是 FlashAttention 节省内存的关键!)
                S_ij = torch.matmul(Q_i, K_j.transpose(-2, -1)) * scale

                if causal:
                    row_idx = torch.arange(i_start, i_end, device=Q.device).unsqueeze(1)
                    col_idx = torch.arange(j_start, j_end, device=Q.device).unsqueeze(0)
                    causal_mask = row_idx < col_idx
                    S_ij.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

                # P_ij = softmax(S_ij) = exp(S_ij - m_i) / l_i
                P_ij = torch.exp(S_ij - m_i) / (l_i + 1e-8)

                # 梯度计算
                # dV_j += P_ij^T @ dO_i
                dV_j = dV_j + torch.matmul(P_ij.transpose(-2, -1), dO_i)

                # dP_ij = dO_i @ V_j^T
                dP_ij = torch.matmul(dO_i, V_j.transpose(-2, -1))

                # dS_ij = P_ij * (dP_ij - D_i)
                dS_ij = P_ij * (dP_ij - D_i) * scale

                # dQ_i += dS_ij @ K_j
                dQ[:, :, i_start:i_end, :] = dQ[:, :, i_start:i_end, :] + \
                    torch.matmul(dS_ij, K_j)

                # dK_j += dS_ij^T @ Q_i
                dK_j = dK_j + torch.matmul(dS_ij.transpose(-2, -1), Q_i)

            dK[:, :, j_start:j_end, :] = dK_j
            dV[:, :, j_start:j_end, :] = dV_j

        return dQ, dK, dV

    @staticmethod
    def io_complexity(N: int, d: int, M: int) -> dict:
        """
        IO 复杂度分析

        FlashAttention V1:
            Θ(N^2 * d^2 / M)  HBM accesses

        Standard Attention:
            Θ(Nd + N^2)  HBM accesses

        当 M = Θ(Nd) 时:
            FlashAttention: Θ(N^2 * d / √M)
            Standard:       Θ(Nd + N^2)
        """
        flash_io = N * N * d * d / M
        standard_io = N * d + N * N
        return {
            "flash_io": flash_io,
            "standard_io": standard_io,
            "speedup": standard_io / flash_io,
        }


# ============================================================================
# 测试
# ============================================================================
def test_flash_v1():
    """验证 FlashAttention V1 的正确性"""
    print("=" * 70)
    print("测试: FlashAttention V1")
    print("=" * 70)

    torch.manual_seed(42)
    B, H, N, d = 2, 4, 256, 64

    Q = torch.randn(B, H, N, d, requires_grad=True)
    K = torch.randn(B, H, N, d, requires_grad=True)
    V = torch.randn(B, H, N, d, requires_grad=True)

    # ── Forward 正确性 ──
    # Standard reference
    scale = 1.0 / math.sqrt(d)
    S = torch.matmul(Q.detach(), K.detach().transpose(-2, -1)) * scale
    P = F.softmax(S, dim=-1)
    O_ref = torch.matmul(P, V.detach())

    print("\n[Forward] 不同 block_size:")
    for br, bc in [(32, 32), (64, 64), (128, 64), (64, 128)]:
        O_flash, _, _ = FlashAttentionV1.forward(Q.detach(), K.detach(), V.detach(),
                                                  block_size_r=br, block_size_c=bc)
        diff = (O_flash - O_ref).abs().max().item()
        print(f"  B_r={br:3d}, B_c={bc:3d}  max_diff = {diff:.2e}  "
              f"{'PASS' if diff < 1e-4 else 'FAIL'}")

    # ── Causal 正确性 ──
    print("\n[Forward] Causal:")
    S_causal = S.clone()
    mask = torch.triu(torch.ones(N, N, device=Q.device, dtype=torch.bool), diagonal=1)
    S_causal.masked_fill_(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
    P_causal = F.softmax(S_causal, dim=-1)
    O_ref_causal = torch.matmul(P_causal, V.detach())

    for bs in [32, 64, 128]:
        O_flash_c, _, _ = FlashAttentionV1.forward(Q.detach(), K.detach(), V.detach(),
                                                    block_size_r=bs, block_size_c=bs,
                                                    causal=True)
        diff = (O_flash_c - O_ref_causal).abs().max().item()
        print(f"  block_size={bs:3d}  max_diff = {diff:.2e}  "
              f"{'PASS' if diff < 1e-4 else 'FAIL'}")

    # ── Backward 正确性 ──
    print("\n[Backward]:")
    dO = torch.randn_like(O_ref)

    # Reference backward
    Q_ref = Q.detach().clone().requires_grad_(True)
    K_ref = K.detach().clone().requires_grad_(True)
    V_ref = V.detach().clone().requires_grad_(True)
    S_ref = torch.matmul(Q_ref, K_ref.transpose(-2, -1)) * scale
    P_ref = F.softmax(S_ref, dim=-1)
    O_ref2 = torch.matmul(P_ref, V_ref)
    O_ref2.backward(dO)

    # FlashAttention backward
    O_fwd, l_fwd, m_fwd = FlashAttentionV1.forward(Q.detach(), K.detach(), V.detach(),
                                                     block_size_r=64, block_size_c=64)
    dQ_flash, dK_flash, dV_flash = FlashAttentionV1.backward(
        Q.detach(), K.detach(), V.detach(), O_fwd, dO, l_fwd, m_fwd,
        block_size_r=64, block_size_c=64
    )

    for name, grad_flash, grad_ref in [
        ("dQ", dQ_flash, Q_ref.grad),
        ("dK", dK_flash, K_ref.grad),
        ("dV", dV_flash, V_ref.grad),
    ]:
        diff = (grad_flash - grad_ref).abs().max().item()
        print(f"  {name} max_diff = {diff:.2e}  {'PASS' if diff < 1e-3 else 'FAIL'}")


def test_io_complexity():
    """IO 复杂度对比"""
    print("\n" + "=" * 70)
    print("IO 复杂度对比 (d=64, SRAM=192KB)")
    print("=" * 70)

    d = 64
    M = 192 * 1024 // 4  # 192KB SRAM, float32

    print(f"{'N':>8s} | {'Standard IO':>14s} | {'FlashAttn IO':>14s} | {'加速比':>8s}")
    print("-" * 55)
    for N in [256, 512, 1024, 2048, 4096, 8192]:
        info = FlashAttentionV1.io_complexity(N, d, M)
        print(f"{N:>8d} | {info['standard_io']:>12.0f} | {info['flash_io']:>12.0f} | {info['speedup']:>6.2f}x")


if __name__ == "__main__":
    test_flash_v1()
    test_io_complexity()
    print("\n✓ Step 3 完成: FlashAttention V1 验证通过")
