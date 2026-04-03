"""
Step 1: Softmax Evolution — 从 3-Pass Safe-Softmax 到 2-Pass Online-Softmax

这是整个 FlashAttention 技术链路的起点。理解 Softmax 的优化历程
是理解 FlashAttention 的关键前置知识。

==========================================================================
1.1 Naive Softmax (不安全，会溢出)
==========================================================================

    softmax(x_i) = exp(x_i) / Σ_j exp(x_j)

问题: 当 x_i 很大时 (如 x > 88 for float32)，exp(x_i) → inf，产生数值溢出。

==========================================================================
1.2 Three-Pass Safe-Softmax (3次遍历，安全)
==========================================================================

    Pass 1: m = max(x)                    → 遍历一次求全局最大值
    Pass 2: d = Σ_j exp(x_j - m)         → 遍历一次求分母
    Pass 3: softmax(x_i) = exp(x_i - m) / d  → 遍历一次求结果

需要 3 次遍历数据 (3 passes over data)。对于 GPU 来说，每次遍历
都是一次全局内存读取，IO 代价很高。

==========================================================================
1.3 Two-Pass Online-Softmax (2次遍历，Milakov & Gimelshein, 2018)
==========================================================================

核心思想: 将 Pass 1 和 Pass 2 合并为一次遍历，利用递推公式在线更新。

定义递推关系:
    m_j = max(m_{j-1}, x_j)              → 在线更新最大值
    d_j = d_{j-1} * exp(m_{j-1} - m_j) + exp(x_j - m_j)  → 在线更新分母

推导:
    d_j = Σ_{i=1}^{j} exp(x_i - m_j)
        = Σ_{i=1}^{j-1} exp(x_i - m_j) + exp(x_j - m_j)
        = Σ_{i=1}^{j-1} exp(x_i - m_{j-1}) * exp(m_{j-1} - m_j) + exp(x_j - m_j)
        = d_{j-1} * exp(m_{j-1} - m_j) + exp(x_j - m_j)

这样 Pass 1 + Pass 2 合并为一个 Online Pass:
    Online Pass: 遍历一次，同时更新 m 和 d
    Pass 2:      遍历一次，计算 softmax(x_i) = exp(x_i - m_N) / d_N

只需 2 次遍历！

==========================================================================
1.4 One-Pass FlashAttention Softmax (1次遍历!)
==========================================================================

FlashAttention 的关键洞察: 在 Attention 计算中，我们不需要显式存储
softmax 的完整结果！我们可以将 softmax 和 V 的加权求和融合在一起，
实现 1-pass:

    o_j = o_{j-1} * (d_{j-1}/d_j) * exp(m_{j-1} - m_j) 
          + exp(x_j - m_j) / d_j * v_j

其中 o_j 是注意力输出的在线累积值。

这就是 FlashAttention 的数学核心！
"""

import torch
import torch.nn.functional as F
import math
import time
from typing import Tuple


# ============================================================================
# 1.1 Naive Softmax (不安全)
# ============================================================================
def naive_softmax(x: torch.Tensor) -> torch.Tensor:
    """
    朴素 softmax，不做数值稳定化处理。
    当输入值较大时会溢出。

    公式: softmax(x_i) = exp(x_i) / Σ_j exp(x_j)
    """
    exp_x = torch.exp(x)
    return exp_x / exp_x.sum(dim=-1, keepdim=True)


# ============================================================================
# 1.2 Three-Pass Safe-Softmax
# ============================================================================
def safe_softmax_3pass(x: torch.Tensor) -> torch.Tensor:
    """
    三次遍历安全 Softmax:
      Pass 1: 求最大值 m = max(x)
      Pass 2: 求分母 d = Σ exp(x_i - m)
      Pass 3: 求结果 softmax(x_i) = exp(x_i - m) / d

    IO 复杂度: 3N 次内存读取 (N = 序列长度)
    """
    # Pass 1: 遍历一次求 max
    m = x.max(dim=-1, keepdim=True).values

    # Pass 2: 遍历一次求 Σexp(x_i - m)
    exp_x = torch.exp(x - m)
    d = exp_x.sum(dim=-1, keepdim=True)

    # Pass 3: 遍历一次求结果
    return exp_x / d


# ============================================================================
# 1.3 Two-Pass Online-Softmax
# ============================================================================
def online_softmax_2pass(x: torch.Tensor) -> torch.Tensor:
    """
    两次遍历 Online Softmax (Milakov & Gimelshein, 2018):
      Online Pass: 一次遍历，同时在线更新 m 和 d
      Output Pass: 一次遍历，计算最终 softmax 结果

    递推公式:
      m_j = max(m_{j-1}, x_j)
      d_j = d_{j-1} * exp(m_{j-1} - m_j) + exp(x_j - m_j)

    IO 复杂度: 2N 次内存读取 (节省了 1/3)
    """
    N = x.shape[-1]
    batch_shape = x.shape[:-1]

    # Online Pass: 合并 max 和 sum 的计算
    m = torch.full((*batch_shape, 1), float('-inf'), device=x.device, dtype=x.dtype)
    d = torch.zeros((*batch_shape, 1), device=x.device, dtype=x.dtype)

    for j in range(N):
        x_j = x[..., j:j+1]
        m_new = torch.maximum(m, x_j)
        d = d * torch.exp(m - m_new) + torch.exp(x_j - m_new)
        m = m_new

    # Output Pass: 计算最终结果
    result = torch.exp(x - m) / d
    return result


def online_softmax_2pass_vectorized(x: torch.Tensor, block_size: int = 64) -> torch.Tensor:
    """
    向量化版本的 Online Softmax — 按 block 处理而非逐元素。
    这是 FlashAttention tiling 策略的雏形。

    将序列分成 blocks，每个 block 内部用标准方式计算，
    block 之间用 online 递推公式更新。
    """
    N = x.shape[-1]
    batch_shape = x.shape[:-1]

    m = torch.full((*batch_shape, 1), float('-inf'), device=x.device, dtype=x.dtype)
    d = torch.zeros((*batch_shape, 1), device=x.device, dtype=x.dtype)

    # Online Pass (block-wise)
    for start in range(0, N, block_size):
        end = min(start + block_size, N)
        x_block = x[..., start:end]

        # Block 内部的 max
        m_block = x_block.max(dim=-1, keepdim=True).values

        # Update global max
        m_new = torch.maximum(m, m_block)

        # Update running sum: 将旧的 d 缩放到新的 max 下
        d = d * torch.exp(m - m_new) + torch.exp(x_block - m_new).sum(dim=-1, keepdim=True)
        m = m_new

    # Output Pass
    return torch.exp(x - m) / d


# ============================================================================
# 1.4 One-Pass Fused Softmax + Attention (FlashAttention 核心)
# ============================================================================
def one_pass_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                       block_size: int = 64) -> torch.Tensor:
    """
    一次遍历的融合 Softmax + Attention 计算。
    这是 FlashAttention 的数学核心。

    关键递推公式 (对 KV 分块):
      S_block = Q @ K_block^T / sqrt(d)
      m_new = max(m_old, rowmax(S_block))
      P_block = exp(S_block - m_new)
      d_new = d_old * exp(m_old - m_new) + rowsum(P_block)
      O_new = O_old * (d_old * exp(m_old - m_new) / d_new) + P_block @ V_block / d_new

    最终只需要 O(N) 额外内存，不需要存储完整的 N×N 注意力矩阵！

    参数:
        Q: [batch, heads, seq_q, d]
        K: [batch, heads, seq_k, d]
        V: [batch, heads, seq_k, d]
    返回:
        O: [batch, heads, seq_q, d]
    """
    B, H, N_q, d = Q.shape
    N_k = K.shape[2]
    scale = 1.0 / math.sqrt(d)

    # 初始化
    O = torch.zeros_like(Q)
    m = torch.full((B, H, N_q, 1), float('-inf'), device=Q.device, dtype=Q.dtype)
    d_sum = torch.zeros((B, H, N_q, 1), device=Q.device, dtype=Q.dtype)

    # 对 K, V 分块遍历 (one pass over K/V)
    for j_start in range(0, N_k, block_size):
        j_end = min(j_start + block_size, N_k)

        K_block = K[:, :, j_start:j_end, :]  # [B, H, block, d]
        V_block = V[:, :, j_start:j_end, :]  # [B, H, block, d]

        # 计算当前 block 的 attention score
        S_block = torch.matmul(Q, K_block.transpose(-2, -1)) * scale  # [B,H,N_q,block]

        # 当前 block 的行最大值
        m_block = S_block.max(dim=-1, keepdim=True).values  # [B,H,N_q,1]

        # 更新全局 max
        m_new = torch.maximum(m, m_block)

        # 缩放因子: 修正旧的累积值到新的 max 下
        exp_diff = torch.exp(m - m_new)  # [B,H,N_q,1]

        # 当前 block 的 exp(S - m_new)
        P_block = torch.exp(S_block - m_new)  # [B,H,N_q,block]

        # 更新分母
        d_new = d_sum * exp_diff + P_block.sum(dim=-1, keepdim=True)

        # 更新输出:
        #   O_new = O_old * (d_old/d_new * exp(m_old-m_new)) + P_block @ V_block / d_new
        O = O * (d_sum * exp_diff / (d_new + 1e-8)) + torch.matmul(P_block, V_block) / (d_new + 1e-8)

        m = m_new
        d_sum = d_new

    return O


# ============================================================================
# 测试与验证
# ============================================================================
def test_softmax_equivalence():
    """验证所有 softmax 实现的数值一致性"""
    print("=" * 70)
    print("测试: Softmax 变体的数值一致性")
    print("=" * 70)

    torch.manual_seed(42)
    x = torch.randn(2, 4, 128)  # [batch=2, heads=4, seq=128]

    # Reference
    ref = F.softmax(x, dim=-1)

    # 各实现
    results = {
        "3-Pass Safe-Softmax": safe_softmax_3pass(x),
        "2-Pass Online-Softmax (scalar)": online_softmax_2pass(x),
        "2-Pass Online-Softmax (block)": online_softmax_2pass_vectorized(x, block_size=32),
    }

    for name, result in results.items():
        max_diff = (result - ref).abs().max().item()
        print(f"  {name:42s} max_diff = {max_diff:.2e}  {'PASS' if max_diff < 1e-5 else 'FAIL'}")

    # 测试溢出情况
    print("\n测试: 大数值输入 (x ~ 100)")
    x_large = torch.randn(2, 256) * 100
    ref_large = F.softmax(x_large, dim=-1)

    naive_result = naive_softmax(x_large)
    safe_result = safe_softmax_3pass(x_large)
    online_result = online_softmax_2pass_vectorized(x_large)

    has_nan_naive = torch.isnan(naive_result).any().item()
    has_nan_safe = torch.isnan(safe_result).any().item()
    has_nan_online = torch.isnan(online_result).any().item()

    print(f"  Naive Softmax  含 NaN: {has_nan_naive}  ← 预期: True (溢出)")
    print(f"  Safe Softmax   含 NaN: {has_nan_safe}   ← 预期: False")
    print(f"  Online Softmax 含 NaN: {has_nan_online}  ← 预期: False")


def test_one_pass_attention():
    """验证 one-pass attention 与标准实现的数值一致性"""
    print("\n" + "=" * 70)
    print("测试: One-Pass Attention vs 标准 Attention")
    print("=" * 70)

    torch.manual_seed(42)
    B, H, N, d = 2, 4, 128, 64
    Q = torch.randn(B, H, N, d)
    K = torch.randn(B, H, N, d)
    V = torch.randn(B, H, N, d)

    # 标准 attention
    scale = 1.0 / math.sqrt(d)
    S = torch.matmul(Q, K.transpose(-2, -1)) * scale
    P = F.softmax(S, dim=-1)
    O_ref = torch.matmul(P, V)

    # One-pass attention (不同 block_size)
    for bs in [32, 64, 128]:
        O_flash = one_pass_attention(Q, K, V, block_size=bs)
        max_diff = (O_flash - O_ref).abs().max().item()
        print(f"  block_size={bs:3d}  max_diff = {max_diff:.2e}  {'PASS' if max_diff < 1e-4 else 'FAIL'}")


if __name__ == "__main__":
    test_softmax_equivalence()
    test_one_pass_attention()
    print("\n✓ Step 1 完成: Softmax 演进链路验证通过")
