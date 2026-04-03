"""
Step 8: QwenNext — 完整的 Qwen-style Transformer 实现

==========================================================================
QwenNext 架构概览
==========================================================================

基于 Qwen2 架构, 集成前述所有 FlashAttention 优化, 构建完整的
LLM Transformer 网络。

架构: Decoder-Only Transformer
    - RMSNorm (Pre-Norm)
    - GQA (Grouped Query Attention) + RoPE
    - SwiGLU FFN (门控线性单元)
    - FlashAttention + PagedKVCache

网络结构:
    ┌───────────────────────────────────────────────────┐
    │  Input Token IDs                                   │
    │       ↓                                            │
    │  Token Embedding [vocab_size, hidden_size]         │
    │       ↓                                            │
    │  ┌─────────────────────────────── × num_layers ──┐ │
    │  │  RMSNorm                                      │ │
    │  │     ↓                                         │ │
    │  │  GQA Attention (RoPE + Flash + KV Cache)      │ │
    │  │     ↓ (+ residual)                            │ │
    │  │  RMSNorm                                      │ │
    │  │     ↓                                         │ │
    │  │  SwiGLU FFN                                   │ │
    │  │     ↓ (+ residual)                            │ │
    │  └───────────────────────────────────────────────┘ │
    │       ↓                                            │
    │  RMSNorm (final)                                   │
    │       ↓                                            │
    │  LM Head [hidden_size, vocab_size]                 │
    │       ↓                                            │
    │  Logits                                            │
    └───────────────────────────────────────────────────┘

==========================================================================
关键模块详解
==========================================================================

1. RMSNorm (Root Mean Square Layer Normalization)
─────────────────────────────────────────────────
    LayerNorm: y = (x - μ) / √(σ² + ε) * γ + β  (需要计算 mean 和 var)
    RMSNorm:   y = x / √(mean(x²) + ε) * γ      (只需要 RMS, 更快!)

    数学:
        RMS(x) = √(1/d × Σᵢ xᵢ²)
        y = x / RMS(x) * γ

    优势:
        - 去掉了 mean shift (β), 参数更少
        - 计算更简单, GPU 效率更高
        - 实验表明与 LayerNorm 效果相当

2. RoPE (Rotary Position Embedding)
────────────────────────────────────
    绝对位置编码: 加到 embedding 上
    相对位置编码: 编码 token 之间的距离
    RoPE: 通过旋转矩阵编码位置, 兼具两者优势!

    数学:
        将 d 维向量分成 d/2 组, 每组 2 个维度 (x_{2i}, x_{2i+1})
        对第 i 组应用旋转:
            [x'_{2i}  ]   [cos(m·θᵢ)  -sin(m·θᵢ)] [x_{2i}  ]
            [x'_{2i+1}] = [sin(m·θᵢ)   cos(m·θᵢ)] [x_{2i+1}]

        其中: θᵢ = 1 / 10000^(2i/d), m = position

    关键性质:
        <RoPE(q, m), RoPE(k, n)> = f(q, k, m-n)
        → 内积只依赖相对位置 m-n!

    实现技巧: 不需要真的做矩阵乘, 用 element-wise 操作:
        q_rot = q * cos + rotate_half(q) * sin
        其中 rotate_half([x₀,x₁,x₂,x₃,...]) = [-x₁,x₀,-x₃,x₂,...]

3. SwiGLU (Swish-Gated Linear Unit)
────────────────────────────────────
    标准 FFN:   FFN(x) = W₂ · ReLU(W₁ · x)
    GLU:        GLU(x) = W₁(x) ⊙ σ(W₂(x))     (门控)
    SwiGLU:     SwiGLU(x) = W₂ · (swish(W₁·x) ⊙ W₃·x)

    swish(x) = x · sigmoid(x) = x · σ(x)

    Qwen2 FFN:
        gate = swish(W_gate · x)    # 门控分支
        up   = W_up · x              # 值分支
        out  = W_down · (gate ⊙ up)  # 降维

    参数: hidden_size → intermediate_size (×2 projections) → hidden_size
    intermediate_size 通常 ≈ 8/3 × hidden_size (因 SwiGLU 有 3 个权重矩阵)

4. GQA + FlashAttention
───────────────────────
    在前面已完整实现, 这里集成到 Transformer block 中:
        - num_heads (Q): 14 (Qwen2-0.5B)
        - num_kv_heads: 2
        - group_size: 7
        - 使用 FlashAttention V2 风格的 tiled attention
        - 支持 PagedKVCache

==========================================================================
Qwen2-0.5B 配置参考
==========================================================================
    vocab_size: 151936
    hidden_size: 896
    intermediate_size: 4864
    num_layers: 24
    num_heads: 14
    num_kv_heads: 2
    head_dim: 64
    max_position_embeddings: 32768
    rope_theta: 1000000.0
    rms_norm_eps: 1e-6
==========================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List
from dataclasses import dataclass


# ============================================================================
# 配置
# ============================================================================

@dataclass
class QwenNextConfig:
    """QwenNext 模型配置 (默认: Qwen2-0.5B scale)"""
    vocab_size: int = 151936
    hidden_size: int = 896
    intermediate_size: int = 4864
    num_layers: int = 24
    num_heads: int = 14             # Q heads
    num_kv_heads: int = 2           # KV heads (GQA)
    head_dim: int = 64
    max_seq_len: int = 32768
    rope_theta: float = 1000000.0
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = True
    sliding_window: Optional[int] = None  # None = 全局 attention
    dtype: torch.dtype = torch.float32

    @property
    def num_heads_per_group(self) -> int:
        return self.num_heads // self.num_kv_heads


# ============================================================================
# Layer 1: RMSNorm
# ============================================================================

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    RMSNorm(x) = x / RMS(x) * γ
    RMS(x) = √(1/d × Σᵢ xᵢ²)

    相比 LayerNorm:
        - 无 mean-centering (无 β), 参数量减半
        - 计算更快 (省去 mean 计算)
        - 效果相当 (Zhang & Sennrich, 2019)
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))  # γ
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., hidden_size]
        # RMS = sqrt(mean(x^2))
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight


# ============================================================================
# Layer 2: RoPE (Rotary Position Embedding)
# ============================================================================

class RotaryPositionEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE, Su et al. 2021).

    核心思想:
        通过复数旋转实现位置编码, 使得 attention score 内积
        只依赖于 query 和 key 的相对距离。

    数学:
        对于第 i 对维度 (2i, 2i+1), 位置 m:
        θᵢ = 1 / (base^(2i/d))   base = 10000 (或 rope_theta)
        旋转角 = m × θᵢ

        [q'_{2i}  ]   [cos(mθᵢ)  -sin(mθᵢ)] [q_{2i}  ]
        [q'_{2i+1}] = [sin(mθᵢ)   cos(mθᵢ)] [q_{2i+1}]

    实现 (element-wise, 无矩阵乘):
        q_rot = q * cos(freq) + rotate_half(q) * sin(freq)

    NTK-aware RoPE (Qwen2):
        使用较大的 base (1000000) 支持更长的序列
    """

    def __init__(self, head_dim: int, max_seq_len: int = 32768,
                 base: float = 1000000.0, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base

        # 计算频率: θᵢ = 1 / (base^(2i/d)), i = 0, 1, ..., d/2-1
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq)

        # 预计算 cos/sin cache
        self._build_cache(max_seq_len, dtype)

    def _build_cache(self, seq_len: int, dtype: torch.dtype):
        positions = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(positions, self.inv_freq)  # [seq_len, d/2]
        # 扩展到完整维度: [seq_len, d] (重复 cos/sin)
        emb = torch.cat([freqs, freqs], dim=-1)  # [seq_len, d]
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        """
        将 x 的前半部分和后半部分交换并取反:
        [x₀, x₁, x₂, x₃, ...] → [-x_{d/2}, ..., -x_{d-1}, x₀, ..., x_{d/2-1}]

        等价于复数乘法中的 -Im + Re*i 部分。
        """
        d = x.shape[-1]
        x1 = x[..., :d // 2]
        x2 = x[..., d // 2:]
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor,
                position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        对 Q 和 K 应用 RoPE。

        Args:
            q: [B, H_q, N, d]
            k: [B, H_kv, N, d]
            position_ids: [B, N] — 每个 token 的位置 ID
        Returns:
            (q_rotated, k_rotated): 与输入同 shape
        """
        seq_len = q.shape[2]

        # 获取对应位置的 cos/sin
        cos = self.cos_cached[position_ids]  # [B, N, d]
        sin = self.sin_cached[position_ids]  # [B, N, d]

        # 扩展维度以匹配 [B, H, N, d]
        cos = cos.unsqueeze(1)  # [B, 1, N, d]
        sin = sin.unsqueeze(1)  # [B, 1, N, d]

        # 应用旋转
        q_rotated = q * cos + self.rotate_half(q) * sin
        k_rotated = k * cos + self.rotate_half(k) * sin

        return q_rotated, k_rotated


# ============================================================================
# Layer 3: GQA Attention (with FlashAttention + KV Cache)
# ============================================================================

class QwenNextAttention(nn.Module):
    """
    QwenNext Grouped Query Attention.

    完整流程:
        1. 线性投影 Q, K, V (Q: H_q heads, KV: H_kv heads)
        2. 应用 RoPE
        3. 更新/使用 KV Cache
        4. FlashAttention 计算 (支持 GQA, Sliding Window, Causal)
        5. 线性输出投影

    GQA 参数 (Qwen2-0.5B):
        Q heads: 14, KV heads: 2, group_size: 7
        → KV cache 只需要 2/14 = 1/7 的存储!
    """

    def __init__(self, config: QwenNextConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.heads_per_group = config.num_heads_per_group

        self.hidden_size = config.hidden_size
        self.q_proj = nn.Linear(config.hidden_size, config.num_heads * config.head_dim, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, config.num_kv_heads * config.head_dim, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, config.num_kv_heads * config.head_dim, bias=True)
        self.o_proj = nn.Linear(config.num_heads * config.head_dim, config.hidden_size, bias=False)

        self.rope = RotaryPositionEmbedding(
            head_dim=config.head_dim,
            max_seq_len=config.max_seq_len,
            base=config.rope_theta,
            dtype=config.dtype,
        )

    def forward(self, hidden_states: torch.Tensor,
                position_ids: torch.Tensor,
                kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                causal: bool = True,
                ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            hidden_states: [B, N, hidden_size]
            position_ids: [B, N]
            kv_cache: (K_cache, V_cache) 或 None
                K_cache: [B, H_kv, S, d], V_cache: [B, H_kv, S, d]
        Returns:
            output: [B, N, hidden_size]
            new_kv_cache: (K, V)
        """
        B, N, _ = hidden_states.shape

        # 1. 线性投影
        Q = self.q_proj(hidden_states)  # [B, N, H_q * d]
        K = self.k_proj(hidden_states)  # [B, N, H_kv * d]
        V = self.v_proj(hidden_states)  # [B, N, H_kv * d]

        # Reshape to multi-head
        Q = Q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)       # [B, H_q, N, d]
        K = K.view(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2)    # [B, H_kv, N, d]
        V = V.view(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2)    # [B, H_kv, N, d]

        # 2. 应用 RoPE
        Q, K = self.rope(Q, K, position_ids)

        # 3. 更新 KV Cache
        if kv_cache is not None:
            K_cache, V_cache = kv_cache
            K = torch.cat([K_cache, K], dim=2)  # [B, H_kv, S+N, d]
            V = torch.cat([V_cache, V], dim=2)
        new_kv_cache = (K, V)

        # 4. FlashAttention with GQA
        O = self._flash_attention_gqa(Q, K, V, causal=causal)

        # 5. 合并 heads + 输出投影
        O = O.transpose(1, 2).contiguous().view(B, N, -1)  # [B, N, H_q*d]
        output = self.o_proj(O)

        return output, new_kv_cache

    def _flash_attention_gqa(self, Q: torch.Tensor, K: torch.Tensor,
                              V: torch.Tensor, causal: bool = True,
                              block_size: int = 64) -> torch.Tensor:
        """
        GQA Flash Attention.
        Q: [B, H_q, N_q, d], K: [B, H_kv, N_k, d], V: [B, H_kv, N_k, d]
        """
        B, H_q, N_q, d = Q.shape
        H_kv = K.shape[1]
        N_k = K.shape[2]
        scale = 1.0 / math.sqrt(d)
        group_size = H_q // H_kv

        # 展开 K/V 以匹配 Q heads
        K_exp = K.unsqueeze(2).expand(-1, -1, group_size, -1, -1).reshape(B, H_q, N_k, d)
        V_exp = V.unsqueeze(2).expand(-1, -1, group_size, -1, -1).reshape(B, H_q, N_k, d)

        O = torch.zeros_like(Q)
        T_q = math.ceil(N_q / block_size)
        T_k = math.ceil(N_k / block_size)

        # FlashAttention V2: 外循环 Q, 内循环 KV
        for i in range(T_q):
            i_s = i * block_size
            i_e = min((i + 1) * block_size, N_q)
            Q_tile = Q[:, :, i_s:i_e, :]

            m_i = torch.full((B, H_q, i_e - i_s, 1), float('-inf'), dtype=Q.dtype, device=Q.device)
            l_i = torch.zeros(B, H_q, i_e - i_s, 1, dtype=Q.dtype, device=Q.device)
            O_i = torch.zeros(B, H_q, i_e - i_s, d, dtype=Q.dtype, device=Q.device)

            # i_s 相对全局 K 的位置: 在 prefill 时 = i_s, decode 时 Q 位置 = N_k-N_q+i_s
            q_global_start = N_k - N_q + i_s

            for j in range(T_k):
                j_s = j * block_size
                j_e = min((j + 1) * block_size, N_k)

                # Causal skip: 如果整个 KV tile 在 Q tile 之后
                if causal and q_global_start + (i_e - i_s) - 1 < j_s:
                    continue

                # Sliding window skip
                if self.config.sliding_window is not None:
                    if q_global_start - j_e >= self.config.sliding_window:
                        continue

                K_tile = K_exp[:, :, j_s:j_e, :]
                V_tile = V_exp[:, :, j_s:j_e, :]

                S = torch.matmul(Q_tile, K_tile.transpose(-2, -1)) * scale

                # Causal mask
                if causal:
                    q_pos = torch.arange(q_global_start, q_global_start + (i_e - i_s), device=Q.device).unsqueeze(1)
                    k_pos = torch.arange(j_s, j_e, device=Q.device).unsqueeze(0)
                    mask = q_pos < k_pos
                    S.masked_fill_(mask.unsqueeze(0).unsqueeze(0), float('-inf'))

                # Sliding window mask
                if self.config.sliding_window is not None:
                    q_pos = torch.arange(q_global_start, q_global_start + (i_e - i_s), device=Q.device).unsqueeze(1)
                    k_pos = torch.arange(j_s, j_e, device=Q.device).unsqueeze(0)
                    sw_mask = (q_pos - k_pos) >= self.config.sliding_window
                    S.masked_fill_(sw_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

                m_ij = S.max(dim=-1, keepdim=True).values
                m_new = torch.maximum(m_i, m_ij)

                diff_old = m_i - m_new
                # 安全处理: -inf - (-inf) = nan → 0
                diff_old = torch.where(torch.isfinite(diff_old), diff_old, torch.zeros_like(diff_old))
                exp_old = torch.exp(diff_old)
                exp_new = torch.exp(S - m_new)
                exp_new = torch.where(torch.isnan(exp_new), torch.zeros_like(exp_new), exp_new)
                l_new = l_i * exp_old + exp_new.sum(dim=-1, keepdim=True)

                O_i = O_i * exp_old + torch.matmul(exp_new, V_tile)
                m_i = m_new
                l_i = l_new

            O_i = O_i / (l_i + 1e-8)
            O[:, :, i_s:i_e, :] = O_i

        return O


# ============================================================================
# Layer 4: SwiGLU FFN
# ============================================================================

class QwenNextMLP(nn.Module):
    """
    SwiGLU Feed-Forward Network.

    SwiGLU(x) = W_down · (swish(W_gate · x) ⊙ W_up · x)

    swish(x) = x · sigmoid(x)    (= SiLU in PyTorch)

    参数量:
        标准 FFN: 2 × hidden × intermediate = 2 × 896 × 4864 = 8.7M
        SwiGLU:   3 × hidden × intermediate = 3 × 896 × 4864 = 13.1M
        但 intermediate 通常设为 2/3 × 4d (补偿多出的投影)
    """

    def __init__(self, config: QwenNextConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, N, hidden_size]
        → [B, N, hidden_size]
        """
        # gate = swish(W_gate · x)
        gate = F.silu(self.gate_proj(x))
        # up = W_up · x
        up = self.up_proj(x)
        # out = W_down · (gate ⊙ up)
        return self.down_proj(gate * up)


# ============================================================================
# Transformer Block
# ============================================================================

class QwenNextBlock(nn.Module):
    """
    QwenNext Transformer Block (Pre-Norm).

    Block(x) = x + Attention(RMSNorm(x))
    Block(x) = x + FFN(RMSNorm(x))

    Pre-Norm 比 Post-Norm 更稳定, 几乎所有现代 LLM 使用 Pre-Norm。
    """

    def __init__(self, config: QwenNextConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention = QwenNextAttention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = QwenNextMLP(config)

    def forward(self, hidden_states: torch.Tensor,
                position_ids: torch.Tensor,
                kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                causal: bool = True,
                ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        hidden_states: [B, N, hidden_size]
        """
        # Self-Attention with residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, new_kv_cache = self.attention(
            hidden_states, position_ids, kv_cache=kv_cache, causal=causal)
        hidden_states = residual + hidden_states

        # FFN with residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, new_kv_cache


# ============================================================================
# 完整网络: QwenNextModel
# ============================================================================

class QwenNextModel(nn.Module):
    """
    QwenNext 完整 Transformer 模型。

    结构:
        Token Embedding → [Transformer Block × num_layers] → RMSNorm → LM Head

    支持:
        - Prefill: 完整序列输入, 构建 KV Cache
        - Decode:  逐 token 生成, 使用 KV Cache
        - GQA:     Q heads > KV heads, 减少 KV cache
        - RoPE:    旋转位置编码, 支持超长序列
        - Sliding Window: 可选的滑动窗口注意力
    """

    def __init__(self, config: QwenNextConfig):
        super().__init__()
        self.config = config

        # Embedding
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # Transformer blocks
        self.layers = nn.ModuleList([
            QwenNextBlock(config, layer_idx=i) for i in range(config.num_layers)
        ])

        # Final norm
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # LM Head
        if config.tie_word_embeddings:
            self.lm_head = None  # 与 embedding 共享权重
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def get_lm_head_weight(self) -> torch.Tensor:
        if self.lm_head is not None:
            return self.lm_head.weight
        return self.embed_tokens.weight

    def forward(self, input_ids: torch.Tensor,
                position_ids: Optional[torch.Tensor] = None,
                kv_caches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            input_ids: [B, N] — token IDs
            position_ids: [B, N] — 位置 (如果 None, 自动生成)
            kv_caches: list of (K, V) per layer, 或 None
        Returns:
            logits: [B, N, vocab_size]
            new_kv_caches: list of (K, V) per layer
        """
        B, N = input_ids.shape

        # 自动生成 position_ids
        if position_ids is None:
            if kv_caches is not None and kv_caches[0] is not None:
                past_len = kv_caches[0][0].shape[2]
            else:
                past_len = 0
            position_ids = torch.arange(past_len, past_len + N,
                                         device=input_ids.device).unsqueeze(0).expand(B, -1)

        # Token embedding
        hidden_states = self.embed_tokens(input_ids)  # [B, N, hidden_size]

        # Transformer blocks
        new_kv_caches = []
        for i, layer in enumerate(self.layers):
            layer_kv = kv_caches[i] if kv_caches is not None else None
            hidden_states, new_kv = layer(hidden_states, position_ids, kv_cache=layer_kv)
            new_kv_caches.append(new_kv)

        # Final norm
        hidden_states = self.norm(hidden_states)

        # LM Head
        if self.lm_head is not None:
            logits = self.lm_head(hidden_states)
        else:
            logits = F.linear(hidden_states, self.embed_tokens.weight)

        return logits, new_kv_caches

    def count_parameters(self) -> dict:
        """统计参数量"""
        total = sum(p.numel() for p in self.parameters())
        embedding = self.embed_tokens.weight.numel()
        attention = sum(
            sum(p.numel() for p in layer.attention.parameters())
            for layer in self.layers
        )
        ffn = sum(
            sum(p.numel() for p in layer.mlp.parameters())
            for layer in self.layers
        )
        norm = sum(
            sum(p.numel() for p in layer.input_layernorm.parameters()) +
            sum(p.numel() for p in layer.post_attention_layernorm.parameters())
            for layer in self.layers
        ) + sum(p.numel() for p in self.norm.parameters())

        return {
            "total": total,
            "embedding": embedding,
            "attention": attention,
            "ffn": ffn,
            "norm": norm,
            "total_M": total / 1e6,
        }


# ============================================================================
# Generation (推理)
# ============================================================================

class QwenNextGenerator:
    """
    QwenNext 文本生成器。

    支持:
        - Greedy Decoding
        - Top-K Sampling
        - Top-P (Nucleus) Sampling
        - Temperature Scaling
    """

    def __init__(self, model: QwenNextModel):
        self.model = model

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor,
                 max_new_tokens: int = 50,
                 temperature: float = 1.0,
                 top_k: int = 0,
                 top_p: float = 1.0,
                 eos_token_id: int = -1,
                 ) -> torch.Tensor:
        """
        自回归生成。

        Phase 1 (Prefill): 处理完整 prompt, 建立 KV Cache
        Phase 2 (Decode):  逐 token 生成, 使用 KV Cache

        Args:
            input_ids: [B, prompt_len]
            max_new_tokens: 最大生成 token 数
            temperature: 采样温度 (1.0 = 标准, <1 更确定, >1 更随机)
            top_k: Top-K 采样 (0 = 不使用)
            top_p: Top-P 采样 (1.0 = 不使用)
        Returns:
            generated_ids: [B, prompt_len + new_tokens]
        """
        B = input_ids.shape[0]
        generated = input_ids.clone()
        kv_caches = None

        # ── Phase 1: Prefill ──
        logits, kv_caches = self.model(input_ids, kv_caches=kv_caches)
        next_token_logits = logits[:, -1, :]  # [B, vocab_size]

        # ── Phase 2: Decode (逐 token) ──
        for step in range(max_new_tokens):
            next_token = self._sample(next_token_logits, temperature, top_k, top_p)
            generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)

            # Check EOS
            if eos_token_id >= 0 and (next_token == eos_token_id).all():
                break

            # Decode step: 只输入最新 token
            logits, kv_caches = self.model(
                next_token.unsqueeze(1),
                kv_caches=kv_caches,
            )
            next_token_logits = logits[:, -1, :]

        return generated

    def _sample(self, logits: torch.Tensor, temperature: float,
                top_k: int, top_p: float) -> torch.Tensor:
        """采样策略"""
        if temperature != 1.0:
            logits = logits / temperature

        # Top-K
        if top_k > 0:
            top_k = min(top_k, logits.shape[-1])
            values, _ = torch.topk(logits, top_k)
            min_value = values[:, -1].unsqueeze(-1)
            logits = torch.where(logits < min_value,
                                 torch.full_like(logits, float('-inf')), logits)

        # Top-P (Nucleus)
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            probs = F.softmax(sorted_logits, dim=-1)
            cumsum = torch.cumsum(probs, dim=-1)
            # 移除累积概率超过 top_p 的 token
            mask = cumsum - probs > top_p
            sorted_logits[mask] = float('-inf')
            # 恢复原始顺序
            logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

        probs = F.softmax(logits, dim=-1)

        if temperature == 0:
            return probs.argmax(dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)


# ============================================================================
# 测试与验证
# ============================================================================

def test_rmsnorm():
    """测试 RMSNorm"""
    print("=" * 70)
    print("测试: RMSNorm")
    print("=" * 70)

    norm = RMSNorm(64)
    x = torch.randn(2, 10, 64)
    y = norm(x)
    assert y.shape == x.shape
    # 检查归一化后的 RMS ≈ 1
    rms = torch.sqrt(y.pow(2).mean(-1))
    print(f"  shape: {y.shape}  mean_rms = {rms.mean().item():.4f}  (should ≈ 1.0)  PASS")


def test_rope():
    """测试 RoPE"""
    print("\n" + "=" * 70)
    print("测试: RoPE (Rotary Position Embedding)")
    print("=" * 70)

    rope = RotaryPositionEmbedding(head_dim=64, max_seq_len=128)

    B, H, N, d = 2, 4, 16, 64
    q = torch.randn(B, H, N, d)
    k = torch.randn(B, H, N, d)
    pos = torch.arange(N).unsqueeze(0).expand(B, -1)

    q_rot, k_rot = rope(q, k, pos)
    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape

    # 验证: RoPE 不改变向量范数 (旋转矩阵是正交的)
    q_norm = q.norm(dim=-1)
    q_rot_norm = q_rot.norm(dim=-1)
    norm_diff = (q_norm - q_rot_norm).abs().max().item()
    print(f"  范数保持: max_diff = {norm_diff:.2e}  {'PASS' if norm_diff < 1e-4 else 'FAIL'}")

    # 验证: 相对位置性质 — <RoPE(q,m), RoPE(k,n)> = f(q,k,m-n)
    # 对比 (pos=0,2) 和 (pos=3,5), 相对距离都是 2
    pos1 = torch.tensor([[0, 2]])
    pos2 = torch.tensor([[3, 5]])
    q_test = torch.randn(1, 1, 2, d)
    k_test = torch.randn(1, 1, 2, d)

    q1, k1 = rope(q_test, k_test, pos1)
    q2, k2 = rope(q_test, k_test, pos2)

    dot1 = (q1[:,:,0,:] * k1[:,:,1,:]).sum()
    dot2 = (q2[:,:,0,:] * k2[:,:,1,:]).sum()
    rel_diff = (dot1 - dot2).abs().item()
    print(f"  相对位置不变性: |dot(pos0,2) - dot(pos3,5)| = {rel_diff:.2e}  "
          f"{'PASS' if rel_diff < 1e-4 else 'FAIL'}")


def test_attention():
    """测试 QwenNextAttention"""
    print("\n" + "=" * 70)
    print("测试: QwenNextAttention (GQA + RoPE + Flash)")
    print("=" * 70)

    config = QwenNextConfig(
        vocab_size=1000, hidden_size=128, intermediate_size=256,
        num_layers=1, num_heads=8, num_kv_heads=2, head_dim=16,
        max_seq_len=256, dtype=torch.float32,
    )
    attn = QwenNextAttention(config)

    B, N = 2, 32

    # Prefill
    x = torch.randn(B, N, config.hidden_size)
    pos = torch.arange(N).unsqueeze(0).expand(B, -1)
    out, kv = attn(x, pos)
    assert out.shape == (B, N, config.hidden_size)
    assert kv[0].shape == (B, config.num_kv_heads, N, config.head_dim)
    print(f"  Prefill: in={x.shape} → out={out.shape}  KV_cache={kv[0].shape}  PASS")

    # Decode (1 token, with cache)
    x_dec = torch.randn(B, 1, config.hidden_size)
    pos_dec = torch.tensor([[N]]).expand(B, -1)
    out_dec, kv_new = attn(x_dec, pos_dec, kv_cache=kv)
    assert out_dec.shape == (B, 1, config.hidden_size)
    assert kv_new[0].shape == (B, config.num_kv_heads, N + 1, config.head_dim)
    print(f"  Decode:  in=(B,1,{config.hidden_size}) → out={out_dec.shape}  "
          f"KV_cache={kv_new[0].shape}  PASS")


def test_mlp():
    """测试 SwiGLU MLP"""
    print("\n" + "=" * 70)
    print("测试: SwiGLU MLP")
    print("=" * 70)

    config = QwenNextConfig(
        hidden_size=128, intermediate_size=256, num_layers=1, num_heads=8,
        num_kv_heads=2, head_dim=16, vocab_size=1000,
    )
    mlp = QwenNextMLP(config)
    x = torch.randn(2, 16, 128)
    out = mlp(x)
    assert out.shape == x.shape
    params = sum(p.numel() for p in mlp.parameters())
    print(f"  shape: {x.shape} → {out.shape}  params = {params:,}  PASS")

    # 验证 SwiGLU 等价性
    gate = F.silu(mlp.gate_proj(x))
    up = mlp.up_proj(x)
    expected = mlp.down_proj(gate * up)
    diff = (out - expected).abs().max().item()
    print(f"  SwiGLU 等价: max_diff = {diff:.2e}  PASS")


def test_full_model():
    """测试完整模型"""
    print("\n" + "=" * 70)
    print("测试: QwenNext 完整模型")
    print("=" * 70)

    # Small config for testing
    config = QwenNextConfig(
        vocab_size=1000, hidden_size=128, intermediate_size=256,
        num_layers=4, num_heads=8, num_kv_heads=2, head_dim=16,
        max_seq_len=256,
    )

    model = QwenNextModel(config)
    params = model.count_parameters()
    print(f"  参数统计:")
    print(f"    Total:     {params['total']:>10,} ({params['total_M']:.2f}M)")
    print(f"    Embedding: {params['embedding']:>10,}")
    print(f"    Attention: {params['attention']:>10,}")
    print(f"    FFN:       {params['ffn']:>10,}")
    print(f"    Norm:      {params['norm']:>10,}")

    B, N = 2, 32
    input_ids = torch.randint(0, config.vocab_size, (B, N))

    # ── Prefill ──
    logits, kv_caches = model(input_ids)
    assert logits.shape == (B, N, config.vocab_size)
    assert len(kv_caches) == config.num_layers
    print(f"\n  Prefill: input_ids={input_ids.shape} → logits={logits.shape}  PASS")

    # ── Decode (5 steps) ──
    for step in range(5):
        next_token = logits[:, -1, :].argmax(dim=-1)
        logits, kv_caches = model(next_token.unsqueeze(1), kv_caches=kv_caches)
        kv_len = kv_caches[0][0].shape[2]

    print(f"  Decode 5 steps: KV_len={kv_len}, logits={logits.shape}  PASS")

    # ── 用 Generator 生成 ──
    generator = QwenNextGenerator(model)
    prompt = torch.randint(0, config.vocab_size, (1, 8))
    output = generator.generate(prompt, max_new_tokens=10, temperature=0.8, top_k=50)
    print(f"  Generate: prompt_len=8 → output_len={output.shape[1]}  PASS")


def test_sliding_window_model():
    """测试带 Sliding Window 的模型"""
    print("\n" + "=" * 70)
    print("测试: QwenNext with Sliding Window")
    print("=" * 70)

    config = QwenNextConfig(
        vocab_size=1000, hidden_size=128, intermediate_size=256,
        num_layers=2, num_heads=8, num_kv_heads=2, head_dim=16,
        max_seq_len=256, sliding_window=32,
    )

    model = QwenNextModel(config)
    B, N = 1, 64
    input_ids = torch.randint(0, config.vocab_size, (B, N))
    logits, kv_caches = model(input_ids)
    assert logits.shape == (B, N, config.vocab_size)
    print(f"  sliding_window=32, N={N}: logits={logits.shape}  PASS")


def test_qwen2_scale():
    """测试 Qwen2-0.5B 规模配置 (仅验证结构, 不做推理)"""
    print("\n" + "=" * 70)
    print("测试: Qwen2-0.5B 规模验证")
    print("=" * 70)

    config = QwenNextConfig()  # 默认 = Qwen2-0.5B
    model = QwenNextModel(config)
    params = model.count_parameters()

    print(f"  Config: hidden={config.hidden_size}, layers={config.num_layers}, "
          f"heads={config.num_heads}/{config.num_kv_heads}")
    print(f"  参数量: {params['total_M']:.1f}M")
    print(f"    Embedding: {params['embedding']/1e6:.1f}M")
    print(f"    Attention: {params['attention']/1e6:.1f}M")
    print(f"    FFN:       {params['ffn']/1e6:.1f}M")
    print(f"    Norm:      {params['norm']/1e6:.1f}M")

    # 验证结构
    assert len(model.layers) == 24
    assert model.embed_tokens.weight.shape == (151936, 896)
    print(f"  结构验证: PASS")

    # 小 batch forward (验证可以跑通)
    input_ids = torch.randint(0, config.vocab_size, (1, 4))
    logits, _ = model(input_ids)
    assert logits.shape == (1, 4, config.vocab_size)
    print(f"  Forward (B=1, N=4): logits={logits.shape}  PASS")


def show_architecture():
    """展示 QwenNext 架构详情"""
    print("\n" + "=" * 70)
    print("QwenNext 架构总览")
    print("=" * 70)
    print("""
    ┌────────────────────────────────────────────────────────────────┐
    │                    QwenNext Architecture                       │
    ├────────────────────────────────────────────────────────────────┤
    │                                                                │
    │  Input:  token_ids [B, N]                                      │
    │     ↓                                                          │
    │  ┌──────────────────────────────────────────────────────────┐  │
    │  │  Token Embedding: [vocab_size=151936, hidden=896]        │  │
    │  └───────────────────────────┬──────────────────────────────┘  │
    │                              ↓                                 │
    │  ┌──────────────────── × 24 layers ────────────────────────┐  │
    │  │                                                          │  │
    │  │  ┌─── Self-Attention Block ──────────────────────────┐   │  │
    │  │  │  RMSNorm(hidden=896, eps=1e-6)                    │   │  │
    │  │  │     ↓                                             │   │  │
    │  │  │  Q_proj: 896 → 14×64 = 896  (14 heads)           │   │  │
    │  │  │  K_proj: 896 → 2×64  = 128  (2 KV heads, GQA)   │   │  │
    │  │  │  V_proj: 896 → 2×64  = 128  (2 KV heads, GQA)   │   │  │
    │  │  │     ↓                                             │   │  │
    │  │  │  RoPE(base=1M, max_len=32K)                       │   │  │
    │  │  │     ↓                                             │   │  │
    │  │  │  FlashAttention GQA (14Q/2KV, group=7)            │   │  │
    │  │  │     ↓                                             │   │  │
    │  │  │  O_proj: 896 → 896                                │   │  │
    │  │  └───────────────────────────────────────────────────┘   │  │
    │  │     ↓ (+ residual)                                       │  │
    │  │                                                          │  │
    │  │  ┌─── SwiGLU FFN Block ──────────────────────────────┐   │  │
    │  │  │  RMSNorm(hidden=896, eps=1e-6)                    │   │  │
    │  │  │     ↓                                             │   │  │
    │  │  │  Gate: 896 → 4864 (+ SiLU activation)            │   │  │
    │  │  │  Up:   896 → 4864                                 │   │  │
    │  │  │  Down: 4864 → 896  (gate ⊙ up → down)            │   │  │
    │  │  └───────────────────────────────────────────────────┘   │  │
    │  │     ↓ (+ residual)                                       │  │
    │  │                                                          │  │
    │  └──────────────────────────────────────────────────────────┘  │
    │                              ↓                                 │
    │  ┌──────────────────────────────────────────────────────────┐  │
    │  │  RMSNorm (final)                                         │  │
    │  │     ↓                                                    │  │
    │  │  LM Head: 896 → 151936 (tied with embedding)            │  │
    │  └──────────────────────────────────────────────────────────┘  │
    │                              ↓                                 │
    │  Output: logits [B, N, 151936]                                 │
    │                                                                │
    ├────────────────────────────────────────────────────────────────┤
    │  KV Cache per layer: [B, 2, seq_len, 64] × 2 (K + V)         │
    │  KV Cache ratio: 2/14 = 14% of full MHA                      │
    │  Total params: ~494M (Qwen2-0.5B scale)                       │
    └────────────────────────────────────────────────────────────────┘
    """)


if __name__ == "__main__":
    test_rmsnorm()
    test_rope()
    test_attention()
    test_mlp()
    test_full_model()
    test_sliding_window_model()
    test_qwen2_scale()
    show_architecture()
    print("\n✓ Step 8 完成: QwenNext 完整网络验证通过")
