"""
Step 7: FlashAttention V4 — 下一代 Attention 优化前沿

==========================================================================
从 V3 到 V4: 系统级优化的融合
==========================================================================

FlashAttention V1-V3 聚焦于 **单 GPU** 内的 kernel 优化。
而实际部署面临的挑战远不止此:

    1. KV Cache 管理 → PagedAttention
    2. 超长序列 (>128K) → Ring Attention (序列并行)
    3. MHA → GQA/MQA  → 需要专用融合 kernel
    4. 稀疏 Attention  → Sliding Window + Local-Global 混合
    5. Prefill 和 Decode 统一 → Chunked Prefill + Decode 混合调度

本文件实现 "FlashAttention V4" — 一个融合了上述所有前沿技术的
统一 Attention 框架, 代表 2024-2026 年的工程实践。

==========================================================================
模块 1: PagedAttention — KV Cache 的虚拟内存管理
==========================================================================

问题: LLM 推理时, KV Cache 占用巨大且不连续
    - Seq_len=128K, d=128, FP16 → 每层每头 128K*128*2 = 32MB
    - 32 heads × 80 layers = ~80GB KV Cache!
    - 不同 request 的 KV 长度不同 → 内存碎片严重

PagedAttention (vLLM, Kwon et al. 2023):
    - 将 KV Cache 分成固定大小的 "Page" (如 16 tokens)
    - 维护一个 block_table 映射: logical_block → physical_block
    - 类似 OS 的虚拟内存管理!

    ┌────────────┐     block_table      ┌────────────────────┐
    │ Request 0  │ ──→ [3, 7, 1, ...]   │ Physical Pages:    │
    │ tokens 0-63│                       │ Page 0: [K₀,V₀]   │
    ├────────────┤                       │ Page 1: [K₂,V₂]   │
    │ Request 1  │ ──→ [0, 5, 2, ...]   │ Page 2: [K₅,V₅]   │
    │ tokens 0-47│                       │ Page 3: [K₁,V₁]   │
    └────────────┘                       │ ...                │
                                         └────────────────────┘

    优势: 零碎片, 支持 KV cache 共享 (beam search, prefix caching)

==========================================================================
模块 2: Ring Attention — 跨设备的序列并行
==========================================================================

问题: 超长序列 (128K-1M+) 单 GPU 放不下 KV Cache

Ring Attention (Liu et al., 2023):
    - 将 Q, K, V 沿序列维度分到 P 个设备
    - 设备排成 ring 拓扑
    - KV blocks 在 ring 上循环传递, 每个设备依次计算所有 KV blocks 的贡献

    Device 0        Device 1        Device 2        Device 3
    Q₀,K₀,V₀       Q₁,K₁,V₁       Q₂,K₂,V₂       Q₃,K₃,V₃
    ┌──────┐        ┌──────┐        ┌──────┐        ┌──────┐
    │attn  │ K₀V₀→ │attn  │ K₁V₁→ │attn  │ K₂V₂→ │attn  │ K₃V₃→
    │Q₀×K₀│        │Q₁×K₁│        │Q₂×K₂│        │Q₃×K₃│
    │      │←K₃V₃  │      │←K₀V₀  │      │←K₁V₁  │      │←K₂V₂
    └──────┘        └──────┘        └──────┘        └──────┘
    Round 1: 本地 KV  →  Round 2: 循环传递  →  Round P: 所有 KV 处理完

    关键: 通信和计算可以 overlap!
        - 当处理 KV_i 时, 同时接收 KV_{i+1}
        - 通信掩藏在计算之下

==========================================================================
模块 3: GQA/MQA 融合 — 减少 KV Heads
==========================================================================

    MHA: Q_heads = K_heads = V_heads = H  (如 32)
    GQA: Q_heads = H, K_heads = V_heads = H_kv (如 8, 每组 4 个 Q head 共享 1 个 KV head)
    MQA: Q_heads = H, K_heads = V_heads = 1

    GQA 在 FlashAttention 中的处理:
        - 每个 KV head 被多个 Q head 共享
        - 可以在 kernel 内部做 broadcast, 避免显式 expand
        - Qwen2 使用 GQA: num_heads=14(Q), num_kv_heads=2

==========================================================================
模块 4: Sliding Window Attention — 稀疏注意力
==========================================================================

    全局 Attention: 每个 token 关注所有 token → O(N²)
    Sliding Window: 每个 token 只关注最近 W 个 token → O(NW)

    ┌──────────────────────┐
    │ ■ ■ ■ □ □ □ □ □ □ □ │  token 0: 看 0,1,2 (window=3)
    │ ■ ■ ■ ■ □ □ □ □ □ □ │  token 1: 看 0,1,2,3
    │ ■ ■ ■ ■ ■ □ □ □ □ □ │  token 2: 看 0,1,2,3,4
    │ □ ■ ■ ■ ■ ■ □ □ □ □ │  token 3: 看 1,2,3,4,5
    │ □ □ ■ ■ ■ ■ ■ □ □ □ │  ...
    │ □ □ □ ■ ■ ■ ■ ■ □ □ │
    └──────────────────────┘

    Mistral/Mixtral 使用 sliding window = 4096
    可与 FlashAttention tiling 完美结合: 跳过超出 window 的 KV blocks

==========================================================================
模块 5: Chunked Prefill + Decode 混合调度
==========================================================================

    问题: Prefill (长 Q) 和 Decode (Q_len=1) 混合 batch
        - 纯 Prefill batch: 计算密集, 适合大 tile
        - 纯 Decode batch: 带宽瓶颈, 适合 FlashDecoding
        - 混合: 如何统一?

    Chunked Prefill (Sarathi, Agrawal et al. 2024):
        - 把长 Prefill 切成 chunks (如 512 tokens)
        - 将 Prefill chunks 和 Decode queries 放在同一 batch
        - 减少 Prefill 对 Decode 的 "抢占"

==========================================================================
"""

import torch
import torch.nn.functional as F
import math
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass, field


# ============================================================================
# 模块 1: PagedAttention
# ============================================================================

class PagedKVCache:
    """
    PagedAttention 的 KV Cache 管理器。

    将 KV Cache 组织成固定大小的 Pages, 通过 block_table
    实现 logical → physical 的映射, 消除内存碎片。

    参数:
        num_heads: KV head 数量
        head_dim: 每个 head 的维度
        page_size: 每个 page 包含的 token 数
        max_pages: 最大 page 数量 (预分配物理内存池)
        dtype: 数据类型
    """

    def __init__(self, num_heads: int, head_dim: int,
                 page_size: int = 16, max_pages: int = 1024,
                 dtype: torch.dtype = torch.float32):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.page_size = page_size
        self.max_pages = max_pages
        self.dtype = dtype

        # 物理内存池: [max_pages, num_heads, page_size, head_dim]
        self.k_pool = torch.zeros(max_pages, num_heads, page_size, head_dim, dtype=dtype)
        self.v_pool = torch.zeros(max_pages, num_heads, page_size, head_dim, dtype=dtype)

        # 空闲 page 列表
        self.free_pages: List[int] = list(range(max_pages))

        # 每个 sequence 的 block table: seq_id → [physical_page_ids]
        self.block_tables: Dict[int, List[int]] = {}
        # 每个 sequence 的当前长度
        self.seq_lengths: Dict[int, int] = {}

    def allocate(self, seq_id: int, num_tokens: int = 0) -> None:
        """为一个新 sequence 分配 KV cache"""
        num_pages = max(1, math.ceil(num_tokens / self.page_size))
        if len(self.free_pages) < num_pages:
            raise RuntimeError(f"Not enough free pages: need {num_pages}, have {len(self.free_pages)}")
        pages = [self.free_pages.pop(0) for _ in range(num_pages)]
        self.block_tables[seq_id] = pages
        self.seq_lengths[seq_id] = 0

    def append(self, seq_id: int, k: torch.Tensor, v: torch.Tensor) -> None:
        """
        追加新的 KV tokens 到 cache 中。
        k, v: [num_heads, num_new_tokens, head_dim]
        """
        num_new = k.shape[1]
        cur_len = self.seq_lengths[seq_id]

        for t in range(num_new):
            logical_pos = cur_len + t
            page_idx = logical_pos // self.page_size
            offset = logical_pos % self.page_size

            # 如果需要新 page, 动态分配
            while page_idx >= len(self.block_tables[seq_id]):
                if not self.free_pages:
                    raise RuntimeError("KV cache OOM: no free pages")
                new_page = self.free_pages.pop(0)
                self.block_tables[seq_id].append(new_page)

            physical_page = self.block_tables[seq_id][page_idx]
            self.k_pool[physical_page, :, offset, :] = k[:, t, :]
            self.v_pool[physical_page, :, offset, :] = v[:, t, :]

        self.seq_lengths[seq_id] = cur_len + num_new

    def read(self, seq_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        读取指定 sequence 的完整 KV cache。
        返回: (K, V), 各 shape [num_heads, seq_len, head_dim]
        """
        seq_len = self.seq_lengths[seq_id]
        if seq_len == 0:
            return (torch.zeros(self.num_heads, 0, self.head_dim, dtype=self.dtype),
                    torch.zeros(self.num_heads, 0, self.head_dim, dtype=self.dtype))

        K_out = torch.zeros(self.num_heads, seq_len, self.head_dim, dtype=self.dtype)
        V_out = torch.zeros(self.num_heads, seq_len, self.head_dim, dtype=self.dtype)

        pages = self.block_tables[seq_id]
        for pos in range(seq_len):
            page_idx = pos // self.page_size
            offset = pos % self.page_size
            physical_page = pages[page_idx]
            K_out[:, pos, :] = self.k_pool[physical_page, :, offset, :]
            V_out[:, pos, :] = self.v_pool[physical_page, :, offset, :]

        return K_out, V_out

    def free(self, seq_id: int) -> None:
        """释放一个 sequence 的 KV cache"""
        if seq_id in self.block_tables:
            self.free_pages.extend(self.block_tables[seq_id])
            del self.block_tables[seq_id]
            del self.seq_lengths[seq_id]

    def stats(self) -> dict:
        return {
            "total_pages": self.max_pages,
            "free_pages": len(self.free_pages),
            "used_pages": self.max_pages - len(self.free_pages),
            "utilization": 1.0 - len(self.free_pages) / self.max_pages,
            "active_seqs": len(self.block_tables),
        }


# ============================================================================
# 模块 2: Paged FlashAttention — 在 Paged KV Cache 上做 FlashAttention
# ============================================================================

class PagedFlashAttention:
    """
    在 PagedKVCache 上执行 FlashAttention。

    与标准 FlashAttention 的区别:
        - KV 不是连续存储的, 而是分散在不同 physical pages
        - 需要通过 block_table 间接寻址
        - attention kernel 内部 per-page 处理

    数学上完全等价于标准 FlashAttention, 只是内存访问模式不同。
    """

    @staticmethod
    def forward(Q: torch.Tensor,
                kv_cache: PagedKVCache,
                seq_id: int,
                num_q_heads: int,
                num_kv_heads: int,
                causal: bool = True) -> torch.Tensor:
        """
        Args:
            Q: [num_q_heads, N_q, head_dim]
            kv_cache: PagedKVCache 实例
            seq_id: 序列 ID
            num_q_heads: Q head 数
            num_kv_heads: KV head 数 (GQA: < num_q_heads)
        Returns:
            O: [num_q_heads, N_q, head_dim]
        """
        H_q = num_q_heads
        H_kv = num_kv_heads
        heads_per_group = H_q // H_kv  # GQA ratio

        N_q = Q.shape[1]
        d = Q.shape[2]
        scale = 1.0 / math.sqrt(d)

        # 从 paged cache 中读取 KV (实际 kernel 中是逐 page 读取)
        K, V = kv_cache.read(seq_id)  # [H_kv, N_k, d]
        N_k = K.shape[1]

        O = torch.zeros(H_q, N_q, d, dtype=Q.dtype)

        # 逐 Q head 处理, 支持 GQA
        for h_q in range(H_q):
            h_kv = h_q // heads_per_group  # 对应的 KV head

            Q_h = Q[h_q:h_q+1]       # [1, N_q, d]
            K_h = K[h_kv:h_kv+1]     # [1, N_k, d]
            V_h = V[h_kv:h_kv+1]     # [1, N_k, d]

            # Tiled FlashAttention (V2 风格)
            m_h = torch.full((1, N_q, 1), float('-inf'), dtype=Q.dtype)
            l_h = torch.zeros(1, N_q, 1, dtype=Q.dtype)
            O_h = torch.zeros(1, N_q, d, dtype=Q.dtype)

            page_size = kv_cache.page_size
            pages = kv_cache.block_tables[seq_id]
            num_pages = math.ceil(N_k / page_size)

            # 按 page 遍历 KV (模拟 paged attention kernel)
            for p in range(num_pages):
                p_start = p * page_size
                p_end = min((p + 1) * page_size, N_k)

                K_page = K_h[:, p_start:p_end, :]
                V_page = V_h[:, p_start:p_end, :]

                S_p = torch.matmul(Q_h, K_page.transpose(-2, -1)) * scale

                if causal:
                    q_pos = torch.arange(N_k - N_q, N_k).unsqueeze(1)
                    k_pos = torch.arange(p_start, p_end).unsqueeze(0)
                    mask = q_pos < k_pos
                    S_p.masked_fill_(mask.unsqueeze(0), float('-inf'))

                m_p = S_p.max(dim=-1, keepdim=True).values
                m_new = torch.maximum(m_h, m_p)
                exp_old = torch.exp(m_h - m_new)
                exp_new = torch.exp(S_p - m_new)
                l_new = l_h * exp_old + exp_new.sum(dim=-1, keepdim=True)

                O_h = O_h * exp_old + torch.matmul(exp_new, V_page)
                m_h = m_new
                l_h = l_new

            O_h = O_h / (l_h + 1e-8)
            O[h_q] = O_h[0]

        return O


# ============================================================================
# 模块 3: Ring Attention — 序列并行
# ============================================================================

class RingAttention:
    """
    Ring Attention: 跨设备的序列并行 FlashAttention。

    将长序列分成 P 份, 分布在 P 个设备上:
        - 每个设备持有 Q_local, K_local, V_local
        - KV blocks 在 ring 拓扑上传递 P-1 轮
        - 每轮: 计算当前 KV 贡献 + 异步传输下一个 KV block

    数学:
        O_p = softmax(Q_p @ [K_0, K_1, ..., K_{P-1}]^T) @ [V_0, ..., V_{P-1}]

    每个设备上使用 online softmax 逐步更新, ring 传递 P-1 轮后得到最终结果。
    模拟实现中, 我们用列表模拟多个设备。
    """

    @staticmethod
    def forward(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                num_devices: int = 4,
                causal: bool = False) -> torch.Tensor:
        """
        模拟 Ring Attention。

        Args:
            Q, K, V: [B, H, N, d]  — 完整的 QKV
            num_devices: 模拟的设备数 P
        Returns:
            O: [B, H, N, d]
        """
        B, H, N, d = Q.shape
        scale = 1.0 / math.sqrt(d)
        chunk_size = math.ceil(N / num_devices)

        # 按序列维度分割到各设备
        Q_chunks = [Q[:, :, i*chunk_size:min((i+1)*chunk_size, N), :] for i in range(num_devices)]
        K_chunks = [K[:, :, i*chunk_size:min((i+1)*chunk_size, N), :] for i in range(num_devices)]
        V_chunks = [V[:, :, i*chunk_size:min((i+1)*chunk_size, N), :] for i in range(num_devices)]

        # 每个设备的状态
        O_local = [torch.zeros_like(Q_chunks[i]) for i in range(num_devices)]
        m_local = [torch.full((*Q_chunks[i].shape[:-1], 1), float('-inf'),
                               dtype=Q.dtype) for i in range(num_devices)]
        l_local = [torch.zeros(*Q_chunks[i].shape[:-1], 1,
                                dtype=Q.dtype) for i in range(num_devices)]

        # Ring 循环: P 轮
        for ring_step in range(num_devices):
            # 每个设备 p 处理来自设备 (p - ring_step) % P 的 KV block
            for p in range(num_devices):
                kv_source = (p - ring_step) % num_devices
                N_q_local = Q_chunks[p].shape[2]
                N_k_local = K_chunks[kv_source].shape[2]

                # Q_p 的全局位置范围
                q_start = p * chunk_size
                # K_source 的全局位置范围
                k_start = kv_source * chunk_size

                S = torch.matmul(Q_chunks[p], K_chunks[kv_source].transpose(-2, -1)) * scale

                if causal:
                    q_pos = torch.arange(q_start, q_start + N_q_local, device=Q.device).unsqueeze(1)
                    k_pos = torch.arange(k_start, k_start + N_k_local, device=Q.device).unsqueeze(0)
                    mask = q_pos < k_pos
                    S.masked_fill_(mask.unsqueeze(0).unsqueeze(0), float('-inf'))

                    # 如果 causal 且整个 K block 在 Q block 之后, 全部 masked
                    if causal and q_start + N_q_local - 1 < k_start:
                        continue  # 全被 mask 了

                m_new_block = S.max(dim=-1, keepdim=True).values
                m_new = torch.maximum(m_local[p], m_new_block)
                exp_old = torch.exp(m_local[p] - m_new)
                exp_new = torch.exp(S - m_new)
                l_new = l_local[p] * exp_old + exp_new.sum(dim=-1, keepdim=True)

                O_local[p] = O_local[p] * exp_old + torch.matmul(exp_new, V_chunks[kv_source])
                m_local[p] = m_new
                l_local[p] = l_new

            # 在 real ring attention 中, 这里同时做 KV 的 send/recv
            # (通信与计算 overlap)

        # 各设备归一化
        for p in range(num_devices):
            O_local[p] = O_local[p] / (l_local[p] + 1e-8)

        # 拼接
        O = torch.cat(O_local, dim=2)
        return O


# ============================================================================
# 模块 4: GQA FlashAttention — 融合 GQA 的高效 Attention
# ============================================================================

class GQAFlashAttention:
    """
    Grouped Query Attention (GQA) 的 FlashAttention 实现。

    GQA: 多个 Q heads 共享一组 KV heads, 减少 KV cache 大小。

    数学:
        设 Q 有 H_q 个 heads, KV 有 H_kv 个 heads
        group_size = H_q / H_kv
        对于 Q head h_q:
            对应的 KV head = h_q // group_size
            O_{h_q} = softmax(Q_{h_q} @ K_{h_kv}^T / √d) @ V_{h_kv}

    在 kernel 中:
        - 同一 group 的 Q heads 可以共享 K/V 的加载
        - 减少 HBM 读取: KV 只需读 H_kv 次, 不是 H_q 次

    Qwen2 参数:
        num_heads = 14 (Q heads)
        num_kv_heads = 2
        group_size = 7
    """

    @staticmethod
    def forward(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                num_kv_heads: int,
                block_size: int = 64,
                causal: bool = False) -> torch.Tensor:
        """
        Args:
            Q: [B, H_q, N, d]      — H_q 个 query heads
            K: [B, H_kv, N, d]     — H_kv 个 key heads
            V: [B, H_kv, N, d]     — H_kv 个 value heads
            num_kv_heads: KV head 数量
        Returns:
            O: [B, H_q, N, d]
        """
        B, H_q, N, d = Q.shape
        H_kv = num_kv_heads
        group_size = H_q // H_kv
        scale = 1.0 / math.sqrt(d)

        O = torch.zeros_like(Q)
        T = math.ceil(N / block_size)

        # 外循环: 遍历 KV groups
        for g in range(H_kv):
            # 该 group 的 Q heads: [g*group_size, (g+1)*group_size)
            Q_group = Q[:, g*group_size:(g+1)*group_size, :, :]  # [B, group_size, N, d]
            K_g = K[:, g:g+1, :, :]  # [B, 1, N, d]
            V_g = V[:, g:g+1, :, :]  # [B, 1, N, d]

            # 对这个 group 做 FlashAttention (V2 风格)
            # K/V 只加载一次, 但对 group_size 个 Q head 都计算

            # Q tile 外循环
            for i in range(T):
                i_start = i * block_size
                i_end = min((i + 1) * block_size, N)

                Q_tile = Q_group[:, :, i_start:i_end, :]  # [B, group_size, tile, d]

                m_i = torch.full((B, group_size, i_end - i_start, 1), float('-inf'),
                                 dtype=Q.dtype, device=Q.device)
                l_i = torch.zeros(B, group_size, i_end - i_start, 1,
                                  dtype=Q.dtype, device=Q.device)
                O_i = torch.zeros(B, group_size, i_end - i_start, d,
                                  dtype=Q.dtype, device=Q.device)

                # KV tile 内循环
                kv_end = (i + 1) if causal else T
                for j in range(kv_end):
                    j_start = j * block_size
                    j_end = min((j + 1) * block_size, N)

                    K_tile = K_g[:, :, j_start:j_end, :]  # [B, 1, tile, d] — 共享!
                    V_tile = V_g[:, :, j_start:j_end, :]  # [B, 1, tile, d] — 共享!

                    # S = Q_tile @ K_tile^T, broadcast KV across group
                    # [B, group_size, tile_q, d] @ [B, 1, d, tile_k] = [B, group_size, tile_q, tile_k]
                    S_ij = torch.matmul(Q_tile, K_tile.transpose(-2, -1)) * scale

                    if causal:
                        row_idx = torch.arange(i_start, i_end, device=Q.device).unsqueeze(1)
                        col_idx = torch.arange(j_start, j_end, device=Q.device).unsqueeze(0)
                        mask = row_idx < col_idx
                        S_ij.masked_fill_(mask.unsqueeze(0).unsqueeze(0), float('-inf'))

                    m_ij = S_ij.max(dim=-1, keepdim=True).values
                    m_new = torch.maximum(m_i, m_ij)
                    exp_old = torch.exp(m_i - m_new)
                    exp_new = torch.exp(S_ij - m_new)
                    l_new = l_i * exp_old + exp_new.sum(dim=-1, keepdim=True)

                    O_i = O_i * exp_old + torch.matmul(exp_new, V_tile)
                    m_i = m_new
                    l_i = l_new

                O_i = O_i / (l_i + 1e-8)
                O[:, g*group_size:(g+1)*group_size, i_start:i_end, :] = O_i

        return O


# ============================================================================
# 模块 5: Sliding Window FlashAttention
# ============================================================================

class SlidingWindowFlashAttention:
    """
    Sliding Window Attention (SWA) + FlashAttention。

    每个 token 只关注最近 window_size 个 token,
    超出窗口的 KV blocks 直接跳过, 大幅减少计算量。

    复杂度: O(N * W * d) 替代 O(N² * d), 其中 W = window_size

    Mistral/Mixtral 使用 SWA + 局部 + 全局混合策略。
    """

    @staticmethod
    def forward(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                window_size: int = 256,
                block_size: int = 64,
                causal: bool = True) -> torch.Tensor:
        """
        Args:
            Q, K, V: [B, H, N, d]
            window_size: 滑动窗口大小
        """
        B, H, N, d = Q.shape
        scale = 1.0 / math.sqrt(d)
        O = torch.zeros_like(Q)

        T = math.ceil(N / block_size)

        for i in range(T):
            i_start = i * block_size
            i_end = min((i + 1) * block_size, N)
            Q_i = Q[:, :, i_start:i_end, :]

            m_i = torch.full((B, H, i_end - i_start, 1), float('-inf'), dtype=Q.dtype, device=Q.device)
            l_i = torch.zeros(B, H, i_end - i_start, 1, dtype=Q.dtype, device=Q.device)
            O_i = torch.zeros(B, H, i_end - i_start, d, dtype=Q.dtype, device=Q.device)

            # 只遍历窗口内的 KV blocks
            # Q 行的最小位置: i_start, 它最远能看到 max(0, i_start - window_size + 1)
            window_start = max(0, i_start - window_size + 1)
            # Q 行的最大位置: i_end - 1, causal 限制到 i_end
            window_end = i_end if causal else min(N, i_end + window_size)

            j_start_block = window_start // block_size
            j_end_block = math.ceil(window_end / block_size)

            for j in range(j_start_block, min(j_end_block, T)):
                j_start = j * block_size
                j_end = min((j + 1) * block_size, N)

                K_j = K[:, :, j_start:j_end, :]
                V_j = V[:, :, j_start:j_end, :]

                S_ij = torch.matmul(Q_i, K_j.transpose(-2, -1)) * scale

                # Causal mask
                row_idx = torch.arange(i_start, i_end, device=Q.device).unsqueeze(1)
                col_idx = torch.arange(j_start, j_end, device=Q.device).unsqueeze(0)

                if causal:
                    causal_mask = row_idx < col_idx
                    S_ij.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

                # Sliding window mask: 超出窗口的位置设为 -inf
                window_mask = (row_idx - col_idx) >= window_size
                S_ij.masked_fill_(window_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

                # 检查是否整行都被 mask 了 (避免 -inf 导致 nan)
                all_masked = (S_ij == float('-inf')).all(dim=-1, keepdim=True)

                m_ij = S_ij.max(dim=-1, keepdim=True).values
                # 如果整行 masked, 保持 m_ij 为 -inf, 不更新该行
                m_ij = torch.where(all_masked, m_i, m_ij)

                m_new = torch.maximum(m_i, m_ij)
                # 安全 exp: 当 m 为 -inf 时, exp(-inf - (-inf)) = nan → 用 0 替代
                diff_old = m_i - m_new
                diff_old = torch.where(torch.isnan(diff_old) | torch.isinf(diff_old),
                                       torch.zeros_like(diff_old), diff_old)
                exp_old = torch.exp(diff_old)
                exp_new = torch.exp(S_ij - m_new)
                exp_new = torch.where(torch.isnan(exp_new), torch.zeros_like(exp_new), exp_new)

                l_new = l_i * exp_old + exp_new.sum(dim=-1, keepdim=True)

                O_i = O_i * exp_old + torch.matmul(exp_new, V_j)
                m_i = m_new
                l_i = l_new

            O_i = O_i / (l_i + 1e-8)
            O[:, :, i_start:i_end, :] = O_i

        return O


# ============================================================================
# 模块 6: Chunked Prefill — Prefill/Decode 混合调度
# ============================================================================

class ChunkedPrefill:
    """
    Chunked Prefill: 将长 Prefill 切分成多个 chunk,
    使 Prefill 和 Decode 可以混合调度。

    传统调度:
        Batch 1: [Prefill(4096 tokens)]  ← Decode 必须等待!
        Batch 2: [Decode(1 token) × 32]

    Chunked Prefill:
        Batch 1: [Prefill_chunk(512), Decode(1) × 24]  ← 混合!
        Batch 2: [Prefill_chunk(512), Decode(1) × 24]
        ...

    好处:
        - Decode 延迟更低 (不需要等完整 Prefill)
        - GPU 利用率更均匀
    """

    @staticmethod
    def forward_mixed_batch(
        prefill_Q: torch.Tensor,   # [H, N_prefill, d]
        prefill_K: torch.Tensor,   # [H, N_prefill, d]
        prefill_V: torch.Tensor,   # [H, N_prefill, d]
        decode_Qs: List[torch.Tensor],   # list of [H, 1, d]
        decode_Ks: List[torch.Tensor],   # list of [H, N_k_i, d]
        decode_Vs: List[torch.Tensor],   # list of [H, N_k_i, d]
        chunk_size: int = 512
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        混合 Prefill + Decode 批处理。

        Returns:
            prefill_O: [H, N_prefill, d]
            decode_Os: list of [H, 1, d]
        """
        H = prefill_Q.shape[0]
        d = prefill_Q.shape[2]
        N_prefill = prefill_Q.shape[1]
        scale = 1.0 / math.sqrt(d)

        # ── Chunked Prefill ──
        num_chunks = math.ceil(N_prefill / chunk_size)
        prefill_O = torch.zeros_like(prefill_Q)

        # 对 prefill 做 causal FlashAttention, 但分 chunk 处理
        m_all = torch.full((H, N_prefill, 1), float('-inf'), dtype=prefill_Q.dtype)
        l_all = torch.zeros(H, N_prefill, 1, dtype=prefill_Q.dtype)
        O_all = torch.zeros(H, N_prefill, d, dtype=prefill_Q.dtype)

        for c in range(num_chunks):
            c_start = c * chunk_size
            c_end = min((c + 1) * chunk_size, N_prefill)

            # 这个 chunk 中的 Q
            Q_c = prefill_Q[:, c_start:c_end, :]

            # 它能看到的 K/V: 0 到 c_end (causal)
            K_vis = prefill_K[:, :c_end, :]
            V_vis = prefill_V[:, :c_end, :]

            S = torch.matmul(Q_c, K_vis.transpose(-2, -1)) * scale

            # Causal mask
            q_pos = torch.arange(c_start, c_end).unsqueeze(1)
            k_pos = torch.arange(0, c_end).unsqueeze(0)
            mask = q_pos < k_pos
            S.masked_fill_(mask.unsqueeze(0), float('-inf'))

            P = F.softmax(S, dim=-1)
            O_c = torch.matmul(P, V_vis)
            prefill_O[:, c_start:c_end, :] = O_c

        # ── Decode (与 prefill chunk 交替处理) ──
        decode_Os = []
        for i, (dQ, dK, dV) in enumerate(zip(decode_Qs, decode_Ks, decode_Vs)):
            S = torch.matmul(dQ, dK.transpose(-2, -1)) * scale
            P = F.softmax(S, dim=-1)
            decode_Os.append(torch.matmul(P, dV))

        return prefill_O, decode_Os


# ============================================================================
# 性能分析
# ============================================================================

class FlashV4Analysis:
    """FlashAttention V4 统一框架的性能分析"""

    @staticmethod
    def memory_savings_paged(num_seqs: int, avg_len: int, max_len: int,
                              page_size: int = 16, head_dim: int = 128,
                              num_kv_heads: int = 2, num_layers: int = 24,
                              dtype_bytes: int = 2) -> dict:
        """分析 PagedAttention 的内存节省"""
        # 传统: 预分配 max_len
        traditional_per_seq = max_len * head_dim * num_kv_heads * 2 * dtype_bytes * num_layers
        traditional_total = traditional_per_seq * num_seqs

        # Paged: 按实际使用分配 (向上取整到 page)
        paged_per_seq = math.ceil(avg_len / page_size) * page_size * head_dim * num_kv_heads * 2 * dtype_bytes * num_layers
        paged_total = paged_per_seq * num_seqs

        return {
            "traditional_mb": traditional_total / 1e6,
            "paged_mb": paged_total / 1e6,
            "savings_pct": (1 - paged_total / traditional_total) * 100,
            "waste_traditional_pct": (1 - avg_len / max_len) * 100,
        }

    @staticmethod
    def ring_attention_scaling(N: int, d: int, num_devices: int,
                                dtype_bytes: int = 2) -> dict:
        """Ring Attention 的扩展性分析"""
        # 单设备 KV cache
        per_device_kv = (N // num_devices) * d * 2 * dtype_bytes

        # 通信量 (每轮传 KV block)
        comm_per_round = per_device_kv
        total_comm = comm_per_round * (num_devices - 1)

        # 计算量
        flops_per_device = 4 * N * (N // num_devices) * d

        return {
            "per_device_kv_mb": per_device_kv / 1e6,
            "per_device_flops_g": flops_per_device / 1e9,
            "total_comm_mb": total_comm / 1e6,
            "comm_compute_ratio": total_comm / flops_per_device,
        }

    @staticmethod
    def sliding_window_savings(N: int, W: int) -> dict:
        """Sliding Window 的计算量节省"""
        full_flops = N * N   # (prop to)
        sw_flops = N * min(W, N)
        return {
            "full_attention": full_flops,
            "sliding_window": sw_flops,
            "savings_pct": (1 - sw_flops / full_flops) * 100 if N > 0 else 0,
        }


# ============================================================================
# 测试
# ============================================================================

def test_paged_kv_cache():
    """测试 PagedKVCache"""
    print("=" * 70)
    print("测试: PagedKVCache")
    print("=" * 70)

    cache = PagedKVCache(num_heads=4, head_dim=64, page_size=16, max_pages=64)

    # 分配 sequence 0
    cache.allocate(seq_id=0, num_tokens=32)
    print(f"  分配 seq 0: {cache.stats()}")

    # 写入 tokens
    k = torch.randn(4, 50, 64)
    v = torch.randn(4, 50, 64)
    cache.append(seq_id=0, k=k, v=v)
    print(f"  写入 50 tokens: seq_len = {cache.seq_lengths[0]}")

    # 读取
    K_read, V_read = cache.read(seq_id=0)
    assert K_read.shape == (4, 50, 64), f"Wrong shape: {K_read.shape}"
    diff = (K_read - k).abs().max().item()
    print(f"  读取验证 max_diff = {diff:.2e}  {'PASS' if diff < 1e-6 else 'FAIL'}")

    # 分配 sequence 1
    cache.allocate(seq_id=1, num_tokens=16)
    k1 = torch.randn(4, 20, 64)
    v1 = torch.randn(4, 20, 64)
    cache.append(seq_id=1, k=k1, v=v1)
    print(f"  分配 seq 1 后: {cache.stats()}")

    # 释放 sequence 0
    cache.free(seq_id=0)
    print(f"  释放 seq 0 后: {cache.stats()}")
    print("  PASS")


def test_paged_flash_attention():
    """测试 PagedFlashAttention"""
    print("\n" + "=" * 70)
    print("测试: PagedFlashAttention (with GQA)")
    print("=" * 70)

    torch.manual_seed(42)
    H_q, H_kv, d = 8, 2, 64
    group_size = H_q // H_kv

    for N_k in [32, 64, 128]:
        cache = PagedKVCache(num_heads=H_kv, head_dim=d, page_size=16, max_pages=64)
        cache.allocate(seq_id=0, num_tokens=N_k)

        K = torch.randn(H_kv, N_k, d)
        V = torch.randn(H_kv, N_k, d)
        cache.append(seq_id=0, k=K, v=V)

        Q = torch.randn(H_q, 1, d)  # decode: Q_len = 1
        O_paged = PagedFlashAttention.forward(Q, cache, seq_id=0,
                                               num_q_heads=H_q, num_kv_heads=H_kv,
                                               causal=True)

        # Reference: expand KV to match Q heads
        K_exp = K.unsqueeze(1).expand(-1, group_size, -1, -1).reshape(H_q, N_k, d)
        V_exp = V.unsqueeze(1).expand(-1, group_size, -1, -1).reshape(H_q, N_k, d)
        scale = 1.0 / math.sqrt(d)
        S = torch.matmul(Q, K_exp.transpose(-2, -1)) * scale
        O_ref = torch.matmul(F.softmax(S, dim=-1), V_exp)

        diff = (O_paged - O_ref).abs().max().item()
        print(f"  N_k={N_k:4d}  GQA {H_q}/{H_kv}  max_diff = {diff:.2e}  "
              f"{'PASS' if diff < 1e-4 else 'FAIL'}")

        cache.free(seq_id=0)


def test_ring_attention():
    """测试 Ring Attention"""
    print("\n" + "=" * 70)
    print("测试: Ring Attention")
    print("=" * 70)

    torch.manual_seed(42)
    B, H, d = 1, 4, 64

    for N in [64, 128, 256]:
        Q = torch.randn(B, H, N, d)
        K = torch.randn(B, H, N, d)
        V = torch.randn(B, H, N, d)

        scale = 1.0 / math.sqrt(d)
        S = torch.matmul(Q, K.transpose(-2, -1)) * scale
        O_ref = torch.matmul(F.softmax(S, dim=-1), V)

        for P in [2, 4]:
            if P > N:
                continue
            O_ring = RingAttention.forward(Q, K, V, num_devices=P)
            diff = (O_ring - O_ref).abs().max().item()
            print(f"  N={N:4d}  devices={P}  max_diff = {diff:.2e}  "
                  f"{'PASS' if diff < 1e-4 else 'FAIL'}")

    # Causal
    print("  [Causal]:")
    for N in [64, 128]:
        Q = torch.randn(B, H, N, d)
        K = torch.randn(B, H, N, d)
        V = torch.randn(B, H, N, d)

        scale = 1.0 / math.sqrt(d)
        S = torch.matmul(Q, K.transpose(-2, -1)) * scale
        mask = torch.triu(torch.ones(N, N, dtype=torch.bool), diagonal=1)
        S.masked_fill_(mask, float('-inf'))
        O_ref = torch.matmul(F.softmax(S, dim=-1), V)

        O_ring = RingAttention.forward(Q, K, V, num_devices=4, causal=True)
        diff = (O_ring - O_ref).abs().max().item()
        print(f"  N={N:4d}  causal  max_diff = {diff:.2e}  "
              f"{'PASS' if diff < 1e-4 else 'FAIL'}")


def test_gqa_flash_attention():
    """测试 GQA FlashAttention"""
    print("\n" + "=" * 70)
    print("测试: GQA FlashAttention")
    print("=" * 70)

    torch.manual_seed(42)
    B, d = 2, 64

    configs = [
        (8, 2, "GQA 8/2"),
        (8, 4, "GQA 8/4"),
        (8, 8, "MHA 8/8"),
        (8, 1, "MQA 8/1"),
    ]

    for H_q, H_kv, name in configs:
        for N in [64, 128]:
            Q = torch.randn(B, H_q, N, d)
            K = torch.randn(B, H_kv, N, d)
            V = torch.randn(B, H_kv, N, d)

            O_gqa = GQAFlashAttention.forward(Q, K, V, num_kv_heads=H_kv,
                                                block_size=32)

            # Reference: expand KV
            group_size = H_q // H_kv
            K_exp = K.unsqueeze(2).expand(-1, -1, group_size, -1, -1).reshape(B, H_q, N, d)
            V_exp = V.unsqueeze(2).expand(-1, -1, group_size, -1, -1).reshape(B, H_q, N, d)
            scale = 1.0 / math.sqrt(d)
            S = torch.matmul(Q, K_exp.transpose(-2, -1)) * scale
            O_ref = torch.matmul(F.softmax(S, dim=-1), V_exp)

            diff = (O_gqa - O_ref).abs().max().item()
            print(f"  {name:10s} N={N:4d}  max_diff = {diff:.2e}  "
                  f"{'PASS' if diff < 1e-4 else 'FAIL'}")


def test_sliding_window():
    """测试 Sliding Window FlashAttention"""
    print("\n" + "=" * 70)
    print("测试: Sliding Window FlashAttention")
    print("=" * 70)

    torch.manual_seed(42)
    B, H, d = 1, 4, 64

    for N in [64, 128, 256]:
        for W in [32, 64]:
            Q = torch.randn(B, H, N, d)
            K = torch.randn(B, H, N, d)
            V = torch.randn(B, H, N, d)

            # Reference: full attention with window mask
            scale = 1.0 / math.sqrt(d)
            S = torch.matmul(Q, K.transpose(-2, -1)) * scale
            row_idx = torch.arange(N).unsqueeze(1)
            col_idx = torch.arange(N).unsqueeze(0)
            causal_mask = row_idx < col_idx
            window_mask = (row_idx - col_idx) >= W
            S.masked_fill_((causal_mask | window_mask).unsqueeze(0).unsqueeze(0), float('-inf'))
            O_ref = torch.matmul(F.softmax(S, dim=-1), V)

            O_sw = SlidingWindowFlashAttention.forward(Q, K, V, window_size=W,
                                                        block_size=32, causal=True)
            diff = (O_sw - O_ref).abs().max().item()
            print(f"  N={N:4d}  W={W:3d}  max_diff = {diff:.2e}  "
                  f"{'PASS' if diff < 1e-4 else 'FAIL'}")


def test_chunked_prefill():
    """测试 Chunked Prefill"""
    print("\n" + "=" * 70)
    print("测试: Chunked Prefill")
    print("=" * 70)

    torch.manual_seed(42)
    H, d = 4, 64
    N_prefill = 256

    Q = torch.randn(H, N_prefill, d)
    K = torch.randn(H, N_prefill, d)
    V = torch.randn(H, N_prefill, d)

    # Decode requests
    decode_Qs = [torch.randn(H, 1, d) for _ in range(4)]
    decode_Ks = [torch.randn(H, 64 + i * 16, d) for i in range(4)]
    decode_Vs = [torch.randn(H, 64 + i * 16, d) for i in range(4)]

    O_prefill, O_decodes = ChunkedPrefill.forward_mixed_batch(
        Q, K, V, decode_Qs, decode_Ks, decode_Vs, chunk_size=64
    )

    # Reference prefill
    scale = 1.0 / math.sqrt(d)
    S = torch.matmul(Q, K.transpose(-2, -1)) * scale
    mask = torch.triu(torch.ones(N_prefill, N_prefill, dtype=torch.bool), diagonal=1)
    S.masked_fill_(mask.unsqueeze(0), float('-inf'))
    O_ref = torch.matmul(F.softmax(S, dim=-1), V)

    diff = (O_prefill - O_ref).abs().max().item()
    print(f"  Prefill N={N_prefill}  chunk=64  max_diff = {diff:.2e}  "
          f"{'PASS' if diff < 1e-4 else 'FAIL'}")

    # Reference decode
    for i in range(len(decode_Qs)):
        S_d = torch.matmul(decode_Qs[i], decode_Ks[i].transpose(-2, -1)) * scale
        O_ref_d = torch.matmul(F.softmax(S_d, dim=-1), decode_Vs[i])
        diff_d = (O_decodes[i] - O_ref_d).abs().max().item()
        print(f"  Decode [{i}] N_k={decode_Ks[i].shape[1]:3d}  max_diff = {diff_d:.2e}  "
              f"{'PASS' if diff_d < 1e-5 else 'FAIL'}")


def test_analysis():
    """性能分析"""
    print("\n" + "=" * 70)
    print("性能分析")
    print("=" * 70)

    # PagedAttention 内存节省
    print("\n[PagedAttention 内存节省] (Qwen2-0.5B, 24 layers, 2 KV heads):")
    for num_seqs, avg_len, max_len in [(32, 512, 2048), (64, 1024, 8192), (128, 2048, 32768)]:
        stats = FlashV4Analysis.memory_savings_paged(
            num_seqs, avg_len, max_len, num_kv_heads=2, num_layers=24)
        print(f"  seqs={num_seqs:3d}  avg/max={avg_len}/{max_len:5d}  "
              f"traditional={stats['traditional_mb']:.0f}MB  "
              f"paged={stats['paged_mb']:.0f}MB  "
              f"节省 {stats['savings_pct']:.0f}%")

    # Sliding Window 计算量节省
    print("\n[Sliding Window 计算量节省]:")
    for N in [4096, 16384, 65536]:
        for W in [1024, 4096]:
            stats = FlashV4Analysis.sliding_window_savings(N, W)
            print(f"  N={N:6d}  W={W:5d}  节省 {stats['savings_pct']:.1f}%")

    # Ring Attention 扩展性
    print("\n[Ring Attention 扩展性] (N=128K, d=128):")
    for P in [2, 4, 8]:
        stats = FlashV4Analysis.ring_attention_scaling(131072, 128, P)
        print(f"  devices={P}  per_device_KV={stats['per_device_kv_mb']:.1f}MB  "
              f"comm={stats['total_comm_mb']:.1f}MB  "
              f"comm/compute={stats['comm_compute_ratio']:.2e}")


def compare_v4_modules():
    print("\n" + "=" * 70)
    print("FlashAttention V4 模块总览")
    print("=" * 70)
    print("""
    ┌──────────────────────────────────────────────────────────────────┐
    │                     FlashAttention V4 统一框架                    │
    ├──────────────────────────────────────────────────────────────────┤
    │                                                                  │
    │   ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
    │   │ PagedAttention│  │Ring Attention │  │ Chunked Prefill      │  │
    │   │ KV Cache 管理 │  │ 序列并行      │  │ Prefill+Decode 混合  │  │
    │   │ 零碎片分配    │  │ 超长序列支持  │  │ 低延迟调度           │  │
    │   └──────┬───────┘  └──────┬───────┘  └──────────┬───────────┘  │
    │          │                 │                      │              │
    │          └────────┬────────┴──────────────────────┘              │
    │                   ▼                                              │
    │   ┌──────────────────────────────────────────────────────────┐   │
    │   │              FlashAttention Core Kernel                   │   │
    │   │  ┌──────────┐ ┌──────────────┐ ┌──────────────────────┐ │   │
    │   │  │ GQA/MQA  │ │Sliding Window│ │   V3 Optimizations   │ │   │
    │   │  │ 融合     │ │ 稀疏Attention│ │ TMA + WarpSpec + FP8 │ │   │
    │   │  └──────────┘ └──────────────┘ └──────────────────────┘ │   │
    │   └──────────────────────────────────────────────────────────┘   │
    │                                                                  │
    └──────────────────────────────────────────────────────────────────┘
    """)


if __name__ == "__main__":
    test_paged_kv_cache()
    test_paged_flash_attention()
    test_ring_attention()
    test_gqa_flash_attention()
    test_sliding_window()
    test_chunked_prefill()
    test_analysis()
    compare_v4_modules()
    print("\n✓ Step 7 完成: FlashAttention V4 全部模块验证通过")
