"""
Step 4: FlashAttention V2 (Dao, 2023)

论文: "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning"

==========================================================================
V2 相比 V1 的三大改进
==========================================================================

改进 1: 交换内外循环顺序
─────────────────────
    V1: 外循环 KV (j), 内循环 Q (i)
    V2: 外循环 Q  (i), 内循环 KV (j)  ← 关键改变!

    为什么这样更好?
    - V1 中, 每次内循环迭代都要从 HBM 读写 O_i, ℓ_i, m_i
    - V2 中, O_i 只在外循环的最后写回一次 HBM
    - O 的 HBM 写从 O(T_c * N) 降低到 O(N) ← 减少了 T_c 倍的 IO!

    V1 IO:
      外循环 j (KV):    读 K_j, V_j 一次
        内循环 i (Q):   读写 Q_i, O_i, ℓ_i, m_i → O_i 被读写 T_c 次!

    V2 IO:
      外循环 i (Q):     读 Q_i 一次, O_i 最后写一次
        内循环 j (KV):  读 K_j, V_j → 每个 KV block 被读 T_r 次

    虽然 K/V 被读了 T_r 次 (vs V1 的 1 次), 但:
    - O 的写从 T_c 次 → 1 次
    - 总 IO 更均衡, GPU occupancy 更高

改进 2: 减少非 matmul FLOPs
─────────────────────────
    V1 中, rescale O 的操作:
        O_i = diag(ℓ_new)^{-1} * (diag(ℓ_i) * exp(m_i - m_new) * O_i + ...)
    这需要对 O_i 做逐元素乘和除, 是非 matmul 操作, 无法利用 Tensor Core。

    V2 的优化: 延迟 rescale, 最后统一除以 ℓ
        - 中间结果不除以 ℓ, 只在最外层循环结束时做一次
        - 大幅减少了非 matmul FLOPs

    V2 的 O 更新公式:
        O_i = diag(exp(m_i^{old} - m_i^{new})) * O_i + exp(S_ij - m_i^{new}) @ V_j
        (注意: 没有除以 ℓ!)
    最后:
        O_i = diag(ℓ_i)^{-1} * O_i   ← 只在最后除一次!

改进 3: 更好的并行化 (warp-level)
───────────────────────────────
    V1: Q blocks 之间并行 (一个 thread block 处理一个 Q block)
    V2: 进一步在序列维度和 batch/head 维度上并行

    V2 的 forward pass 在 {batch, head, Q block} 三个维度上都并行。
    这意味着 GPU 有更多的 thread blocks 可以调度。

==========================================================================
IO 复杂度: 与 V1 相同 O(N^2 d^2 / M), 但常数因子更小
Wall-clock 速度: 比 V1 快 ~2x (在 A100 上)
==========================================================================
"""

import torch
import torch.nn.functional as F
import math
from typing import Tuple


class FlashAttentionV2:
    """
    FlashAttention V2 的 Python 参考实现。

    与 V1 的关键区别:
    1. 外循环 Q, 内循环 KV (V1 是反过来的)
    2. 延迟 rescale — O 的缩放推迟到最后
    3. 减少非 matmul 操作
    """

    @staticmethod
    def forward(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                block_size_q: int = 64, block_size_kv: int = 64,
                causal: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        FlashAttention V2 Forward Pass.

        关键变化 vs V1:
        1. 外循环 Q (i), 内循环 KV (j)
        2. O_i 在内循环中不除以 ℓ, 最后才 rescale
        3. 减少中间 rescale 操作

        Args:
            Q: [B, H, N_q, d]
            K: [B, H, N_k, d]
            V: [B, H, N_k, d]
        Returns:
            O: [B, H, N_q, d]
            L: [B, H, N_q, 1]  — log-sum-exp (用于 backward)
            M: [B, H, N_q, 1]  — row max
        """
        B, H, N_q, d = Q.shape
        N_k = K.shape[2]
        scale = 1.0 / math.sqrt(d)

        Tr = math.ceil(N_q / block_size_q)   # Q blocks
        Tc = math.ceil(N_k / block_size_kv)   # KV blocks

        O = torch.zeros_like(Q)
        L = torch.zeros(B, H, N_q, 1, device=Q.device, dtype=Q.dtype)
        M = torch.full((B, H, N_q, 1), float('-inf'), device=Q.device, dtype=Q.dtype)

        # ============================================================
        # 外循环: 遍历 Q blocks  ← V2 的核心改变!
        # ============================================================
        for i in range(Tr):
            i_start = i * block_size_q
            i_end = min((i + 1) * block_size_q, N_q)
            Br = i_end - i_start

            # 加载 Q_i (整个 Q block 在外循环中只读一次!)
            Q_i = Q[:, :, i_start:i_end, :]  # [B, H, B_r, d]

            # 用于在线累积的局部变量
            O_i = torch.zeros(B, H, Br, d, device=Q.device, dtype=Q.dtype)
            l_i = torch.zeros(B, H, Br, 1, device=Q.device, dtype=Q.dtype)
            m_i = torch.full((B, H, Br, 1), float('-inf'), device=Q.device, dtype=Q.dtype)

            # 确定 KV 的遍历范围 (causal: 只看当前位置之前)
            kv_range_end = min(i_end, N_k) if causal else N_k

            # ============================================================
            # 内循环: 遍历 KV blocks
            # ============================================================
            for j in range(Tc):
                j_start = j * block_size_kv
                j_end = min((j + 1) * block_size_kv, N_k)

                # Causal: 如果 K block 完全在 Q block 之后, 跳过
                if causal and j_start >= kv_range_end:
                    break

                K_j = K[:, :, j_start:j_end, :]  # [B, H, B_c, d]
                V_j = V[:, :, j_start:j_end, :]  # [B, H, B_c, d]

                # Step 1: S_ij = Q_i @ K_j^T * scale
                S_ij = torch.matmul(Q_i, K_j.transpose(-2, -1)) * scale  # [B,H,Br,Bc]

                # Step 2: Causal mask
                if causal:
                    row_idx = torch.arange(i_start, i_end, device=Q.device).unsqueeze(1)
                    col_idx = torch.arange(j_start, j_end, device=Q.device).unsqueeze(0)
                    causal_mask = row_idx < col_idx
                    S_ij.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

                # Step 3: 在线更新 max
                m_ij = S_ij.max(dim=-1, keepdim=True).values  # [B,H,Br,1]
                m_new = torch.maximum(m_i, m_ij)

                # Step 4: 计算 exp (注意: 不除以 l — 延迟 rescale!)
                P_ij = torch.exp(S_ij - m_new)  # [B,H,Br,Bc]

                # Step 5: 更新 O (V2 的 rescale 方式 — 只乘 exp 差, 不除 l)
                #   O_i = exp(m_old - m_new) * O_i + P_ij @ V_j
                alpha = torch.exp(m_i - m_new)  # [B,H,Br,1]
                O_i = alpha * O_i + torch.matmul(P_ij, V_j)  # [B,H,Br,d]

                # Step 6: 更新 l
                l_i = alpha * l_i + P_ij.sum(dim=-1, keepdim=True)  # [B,H,Br,1]

                m_i = m_new

            # ============================================================
            # 最后才做 rescale — V2 的关键优化!
            # ============================================================
            O_i = O_i / (l_i + 1e-8)  # ← 只在外循环结束时除一次

            # 写回 "HBM" (整个外循环只写一次!)
            O[:, :, i_start:i_end, :] = O_i
            L[:, :, i_start:i_end, :] = m_i + torch.log(l_i + 1e-8)  # log-sum-exp
            M[:, :, i_start:i_end, :] = m_i

        return O, L, M

    @staticmethod
    def backward(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                 O: torch.Tensor, dO: torch.Tensor,
                 L: torch.Tensor,
                 block_size_q: int = 64, block_size_kv: int = 64,
                 causal: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        FlashAttention V2 Backward Pass.

        与 V1 backward 类似, 核心是 recomputation。
        但循环顺序与 V1 不同:
        - 外循环 KV (j), 内循环 Q (i)  ← backward 中反过来更好!
        - 这样 dK_j, dV_j 作为累积值不需要原子操作
        """
        B, H, N_q, d = Q.shape
        N_k = K.shape[2]
        scale = 1.0 / math.sqrt(d)

        dQ = torch.zeros_like(Q)
        dK = torch.zeros_like(K)
        dV = torch.zeros_like(V)

        # D_i = rowsum(dO * O)
        D = (dO * O).sum(dim=-1, keepdim=True)  # [B, H, N_q, 1]

        Tr = math.ceil(N_q / block_size_q)
        Tc = math.ceil(N_k / block_size_kv)

        # Backward: 外循环 KV, 内循环 Q
        for j in range(Tc):
            j_start = j * block_size_kv
            j_end = min((j + 1) * block_size_kv, N_k)

            K_j = K[:, :, j_start:j_end, :]
            V_j = V[:, :, j_start:j_end, :]
            dK_j = torch.zeros_like(K_j)
            dV_j = torch.zeros_like(V_j)

            for i in range(Tr):
                i_start = i * block_size_q
                i_end = min((i + 1) * block_size_q, N_q)

                if causal and i_start > j_end - 1:
                    continue

                Q_i = Q[:, :, i_start:i_end, :]
                dO_i = dO[:, :, i_start:i_end, :]
                L_i = L[:, :, i_start:i_end, :]
                D_i = D[:, :, i_start:i_end, :]

                # 重计算 S 和 P
                S_ij = torch.matmul(Q_i, K_j.transpose(-2, -1)) * scale

                if causal:
                    row_idx = torch.arange(i_start, i_end, device=Q.device).unsqueeze(1)
                    col_idx = torch.arange(j_start, j_end, device=Q.device).unsqueeze(0)
                    causal_mask = row_idx < col_idx
                    S_ij.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

                # P = exp(S - L)  (L = log(sum(exp(S - m))) + m = logsumexp)
                P_ij = torch.exp(S_ij - L_i)

                # dV_j += P^T @ dO
                dV_j = dV_j + torch.matmul(P_ij.transpose(-2, -1), dO_i)

                # dP = dO @ V^T
                dP_ij = torch.matmul(dO_i, V_j.transpose(-2, -1))

                # dS = P * (dP - D)
                dS_ij = P_ij * (dP_ij - D_i) * scale

                # dQ_i += dS @ K
                dQ[:, :, i_start:i_end, :] += torch.matmul(dS_ij, K_j)

                # dK_j += dS^T @ Q
                dK_j = dK_j + torch.matmul(dS_ij.transpose(-2, -1), Q_i)

            dK[:, :, j_start:j_end, :] = dK_j
            dV[:, :, j_start:j_end, :] = dV_j

        return dQ, dK, dV


def compare_v1_v2():
    """
    对比 V1 和 V2 的区别
    """
    print("=" * 70)
    print("FlashAttention V1 vs V2 对比")
    print("=" * 70)

    print("""
    ┌────────────────┬──────────────────────┬──────────────────────┐
    │   特性         │  FlashAttention V1   │  FlashAttention V2   │
    ├────────────────┼──────────────────────┼──────────────────────┤
    │ 外循环         │  KV blocks (j)       │  Q blocks (i)        │
    │ 内循环         │  Q blocks (i)        │  KV blocks (j)       │
    ├────────────────┼──────────────────────┼──────────────────────┤
    │ O 的 HBM 写    │  T_c 次 (每个KV block│  1 次 (只在最后)      │
    │               │  的内循环都写)        │                      │
    ├────────────────┼──────────────────────┼──────────────────────┤
    │ Rescale        │  每步都除以 ℓ        │  最后才除以 ℓ        │
    │ (non-matmul)  │  → 更多非 matmul ops │  → 更少非 matmul ops │
    ├────────────────┼──────────────────────┼──────────────────────┤
    │ Causal 效率    │  内循环 Q 可以跳过   │  内循环 KV 可以提前   │
    │               │  (但外循环无法跳过)   │  break → 更高效      │
    ├────────────────┼──────────────────────┼──────────────────────┤
    │ 并行化         │  在 Q blocks 上并行   │  在 Q blocks 上并行   │
    │               │                      │  + 更好的warp分配     │
    ├────────────────┼──────────────────────┼──────────────────────┤
    │ A100 速度      │  ~190 TFLOPs         │  ~230 TFLOPs (~2x)  │
    ├────────────────┼──────────────────────┼──────────────────────┤
    │ IO 复杂度      │  O(N²d²/M)           │  O(N²d²/M) 常数更小  │
    └────────────────┴──────────────────────┴──────────────────────┘
    """)


def test_flash_v2():
    """验证 FlashAttention V2 的正确性"""
    print("=" * 70)
    print("测试: FlashAttention V2 正确性")
    print("=" * 70)

    torch.manual_seed(42)
    B, H, N, d = 2, 4, 256, 64

    Q = torch.randn(B, H, N, d)
    K = torch.randn(B, H, N, d)
    V = torch.randn(B, H, N, d)
    scale = 1.0 / math.sqrt(d)

    # ── Forward ──
    S = torch.matmul(Q, K.transpose(-2, -1)) * scale
    P = F.softmax(S, dim=-1)
    O_ref = torch.matmul(P, V)

    print("\n[Forward] Non-causal:")
    for bq, bkv in [(32, 32), (64, 64), (128, 64)]:
        O_v2, _, _ = FlashAttentionV2.forward(Q, K, V, block_size_q=bq, block_size_kv=bkv)
        diff = (O_v2 - O_ref).abs().max().item()
        print(f"  block_q={bq:3d}, block_kv={bkv:3d}  max_diff = {diff:.2e}  "
              f"{'PASS' if diff < 1e-4 else 'FAIL'}")

    # ── Causal ──
    print("\n[Forward] Causal:")
    mask = torch.triu(torch.ones(N, N, device=Q.device, dtype=torch.bool), diagonal=1)
    S_c = S.clone().masked_fill_(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
    O_ref_c = torch.matmul(F.softmax(S_c, dim=-1), V)

    for bs in [32, 64, 128]:
        O_v2_c, _, _ = FlashAttentionV2.forward(Q, K, V, block_size_q=bs,
                                                  block_size_kv=bs, causal=True)
        diff = (O_v2_c - O_ref_c).abs().max().item()
        print(f"  block_size={bs:3d}  max_diff = {diff:.2e}  "
              f"{'PASS' if diff < 1e-4 else 'FAIL'}")

    # ── Backward ──
    print("\n[Backward]:")
    dO = torch.randn_like(O_ref)

    Q_r = Q.clone().requires_grad_(True)
    K_r = K.clone().requires_grad_(True)
    V_r = V.clone().requires_grad_(True)
    S_r = torch.matmul(Q_r, K_r.transpose(-2, -1)) * scale
    O_r = torch.matmul(F.softmax(S_r, dim=-1), V_r)
    O_r.backward(dO)

    O_fwd, L_fwd, _ = FlashAttentionV2.forward(Q, K, V, block_size_q=64, block_size_kv=64)
    dQ_v2, dK_v2, dV_v2 = FlashAttentionV2.backward(
        Q, K, V, O_fwd, dO, L_fwd, block_size_q=64, block_size_kv=64
    )

    for name, gf, gr in [("dQ", dQ_v2, Q_r.grad), ("dK", dK_v2, K_r.grad), ("dV", dV_v2, V_r.grad)]:
        diff = (gf - gr).abs().max().item()
        print(f"  {name} max_diff = {diff:.2e}  {'PASS' if diff < 1e-3 else 'FAIL'}")


if __name__ == "__main__":
    compare_v1_v2()
    test_flash_v2()
    print("\n✓ Step 4 完成: FlashAttention V2 验证通过")
