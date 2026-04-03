# FlashAttention 全链路实现 + Qwen3-Next 网络架构

从 Softmax 优化到 FlashAttention V1/V2/V3/V4 再到 FlashDecoding/FlashDecoding++，集成 PagedAttention、Ring Attention、GQA，最终构建 QwenNext (Qwen2) 和 Qwen3-Next 完整 Transformer 网络。

## 目录

| 文件 | 内容 | 关键知识点 |
|------|------|-----------|
| `01_softmax_evolution.py` | Softmax 优化演进 | 3-pass → 2-pass Online-Softmax |
| `02_memory_efficient_attention.py` | 内存高效 Attention | O(N²) → O(N) 显存, tiling |
| `03_flash_attention_v1.py` | FlashAttention V1 | IO-aware, tiling, recomputation |
| `04_flash_attention_v2.py` | FlashAttention V2 | 循环交换, 减少非 matmul FLOPs, 并行度 |
| `05_flash_decoding.py` | FlashDecoding & FlashDecoding++ | KV 并行, 统一 max, decode 优化 |
| `06_flash_attention_v3.py` | FlashAttention V3 | Hopper TMA, warp specialization, FP8 |
| `07_flash_attention_v4.py` | FlashAttention V4 | PagedKV, Ring Attention, GQA, Sliding Window, Chunked Prefill |
| `08_qwen_next.py` | QwenNext 完整网络 | RMSNorm, RoPE, GQA Attention, SwiGLU FFN, KV Cache, 生成 |
| `09_qwen3_next.py` | **Qwen3-Next** 下一代架构 | Hybrid Attention (DeltaNet:Softmax=3:1), Gated Attention, ZC-RMSNorm, MTP 投机采样, High-Sparsity MoE |
| `run_all_tests.py` | 全链路测试 | 正确性验证 + 性能基准 |

## 技术演进路线

```
┌────────────────────────────────────────────────────────────────────────┐
│                          Softmax 优化                                  │
│   Naive Softmax → Safe Softmax (3-pass) → Online Softmax (2-pass)     │
└──────────────────────────────┬─────────────────────────────────────────┘
                               │ 2-pass online 技巧
                               ▼
┌────────────────────────────────────────────────────────────────────────┐
│                       标准 Attention                                   │
│           S = QK^T (O(N²) 显存) → P = softmax(S) → O = PV            │
└──────────────────────────────┬─────────────────────────────────────────┘
                               │ tiling + 不存 S 矩阵
                               ▼
┌────────────────────────────────────────────────────────────────────────┐
│                  Memory Efficient Attention                            │
│       分块计算, O(N) 显存, 但需要 online softmax 校正                    │
└──────────────────────────────┬─────────────────────────────────────────┘
                               │ IO-aware + SRAM tiling
                               ▼
┌────────────────────────────────────────────────────────────────────────┐
│                     FlashAttention V1 (2022)                           │
│  IO 复杂度 O(N²d²/M), tiling in SRAM, forward + backward recompute    │
└──────────────────────────────┬─────────────────────────────────────────┘
                               │ 循环交换 + 减少非 matmul 操作
                               ▼
┌────────────────────────────────────────────────────────────────────────┐
│                     FlashAttention V2 (2023)                           │
│  外循环 Q (非 KV), ~2x 加速, 更好的 seq 维度并行                        │
└──────┬───────────────────────┴───────────────────────┬─────────────────┘
       │ 针对 decode 场景                                │ 针对 Hopper GPU
       ▼                                               ▼
┌────────────────────────┐              ┌────────────────────────────────┐
│   FlashDecoding (2023) │              │    FlashAttention V3 (2024)    │
│  KV split 并行          │              │  TMA + Warp Specialization     │
│  解决 Q_len=1 利用率低   │              │  Ping-Pong Scheduling          │
└──────────┬─────────────┘              │  FP8 支持, ~1.5-2x over V2    │
           │ 统一 max, 减少 sync        └──────────────┬─────────────────┘
           ▼                                           │
┌────────────────────────┐                             │ 系统级融合
│ FlashDecoding++ (2024) │                             ▼
│  Unified Max φ          │              ┌────────────────────────────────┐
│  单 kernel, no reduce   │              │   FlashAttention V4 (统一框架) │
│  Flat GEMM 优化         │              │  PagedAttention (KV Cache)     │
└────────────────────────┘              │  Ring Attention (序列并行)      │
                                         │  GQA/MQA Fused Kernel         │
                                         │  Sliding Window Attention      │
                                         │  Chunked Prefill               │
                                         └──────────────┬─────────────────┘
                                                        │ 集成到完整网络
                                                        ▼
                                         ┌────────────────────────────────┐
                                         │     QwenNext Transformer       │
                                         │  RMSNorm + RoPE + GQA + SwiGLU│
                                         │  Prefill/Decode + KV Cache     │
                                         │  ~494M params (Qwen2-0.5B)    │
                                         └──────────────┬─────────────────┘
                                                        │ 5 大架构升级
                                                        ▼
                                         ┌────────────────────────────────┐
                                         │      Qwen3-Next Transformer    │
                                         │ ① Hybrid Attn (DeltaNet:S=3:1)│
                                         │ ② Gated Attn (消除 Sink)       │
                                         │ ③ Zero-Centered RMSNorm        │
                                         │ ④ MTP (投机采样 1.5-3x 加速)   │
                                         │ ⑤ High-Sparsity MoE (6.25%)   │
                                         └────────────────────────────────┘
```

## 核心数学

### Online Softmax (Milakov & Gimelshein, 2018)

标准 Safe-Softmax 需要 **3 次遍历**:
1. `m = max(x)`
2. `d = Σ exp(x_i - m)`
3. `y_i = exp(x_i - m) / d`

Online Softmax 合并前两次为 **2 次遍历**, 递推公式:

$$m_j = \max(m_{j-1}, x_j)$$

$$d_j = d_{j-1} \cdot \exp(m_{j-1} - m_j) + \exp(x_j - m_j)$$

### FlashAttention 的 Tiled Online Softmax

对于 Attention: $O = \text{softmax}(QK^T / \sqrt{d}) \cdot V$

将 K, V 分块处理, 每处理一个新块 j 时更新:

$$m^{(j)} = \max(m^{(j-1)}, \text{rowmax}(S_{:,j}))$$

$$\ell^{(j)} = \ell^{(j-1)} \cdot \exp(m^{(j-1)} - m^{(j)}) + \text{rowsum}(\exp(S_{:,j} - m^{(j)}))$$

$$O^{(j)} = O^{(j-1)} \cdot \exp(m^{(j-1)} - m^{(j)}) + \exp(S_{:,j} - m^{(j)}) \cdot V_j$$

最终: $O = O^{(T_c)} / \ell^{(T_c)}$

### FlashDecoding: Split-KV Reduce

将 KV 分成 S 份, 每份独立计算 $(O_s, m_s, \ell_s)$, 然后归约:

$$m_{\text{final}} = \max_s(m_s)$$

$$\ell_{\text{final}} = \sum_s \ell_s \cdot \exp(m_s - m_{\text{final}})$$

$$O_{\text{final}} = \frac{\sum_s \ell_s \cdot \exp(m_s - m_{\text{final}}) \cdot O_s}{\ell_{\text{final}}}$$

### FlashDecoding++: Unified Max

使用预估的全局最大值 $\phi$, 使 reduce 无需 max 校正:

$$O_{\text{final}} = \frac{\sum_s O_s}{\sum_s \ell_s} \quad \text{where all splits use same } \phi$$

## IO 复杂度对比

| 算法 | HBM 读取 | HBM 写入 | 显存占用 |
|------|---------|---------|---------|
| Standard Attention | $O(Nd + N^2)$ | $O(Nd + N^2)$ | $O(N^2)$ |
| FlashAttention V1/V2 | $O(N^2d^2/M)$ | $O(Nd)$ | $O(N)$ |
| FlashDecoding | 同上 | $O(Nd)$ | $O(N)$ |

其中 $M$ = SRAM 大小, $N$ = 序列长度, $d$ = head dimension。

## 快速开始

```bash
# 安装依赖
pip install torch numpy

# 运行所有测试 (Step 1-9)
python run_all_tests.py

# 快速测试
python run_all_tests.py --quick

# 带性能基准
python run_all_tests.py --benchmark

# 只跑某一步
python run_all_tests.py --step 1   # Softmax
python run_all_tests.py --step 3   # FlashAttention V1
python run_all_tests.py --step 5   # FlashDecoding
python run_all_tests.py --step 7   # FlashAttention V4
python run_all_tests.py --step 8   # QwenNext
python run_all_tests.py --step 9   # Qwen3-Next

# 单独运行每个文件 (自带测试)
python 01_softmax_evolution.py
python 02_memory_efficient_attention.py
python 03_flash_attention_v1.py
python 04_flash_attention_v2.py
python 05_flash_decoding.py
python 06_flash_attention_v3.py
python 07_flash_attention_v4.py
python 08_qwen_next.py
python 09_qwen3_next.py
```

## 环境要求

- Python >= 3.8
- PyTorch >= 1.12
- NumPy

> **注意**: 本项目使用 PyTorch 模拟 FlashAttention 的算法逻辑, 不依赖 GPU。
> 真正的 GPU kernel 需要 Triton 或 CUDA C++ 实现。

## 各版本核心特性

| 版本 | 年份 | 核心优化 | 目标 GPU | 相比上代加速 |
|------|------|---------|---------|------------|
| V1 | 2022 | IO-aware tiling, recomputation | A100 | ~3-4x vs Standard |
| V2 | 2023 | 循环交换, 减少非matmul FLOPs | A100 | ~2x vs V1 |
| V3 | 2024 | TMA, warp specialization, FP8 | H100 | ~1.5-2x vs V2 |
| V4 (系统级) | 2024-25 | PagedKV, Ring, GQA, SlidingWindow, ChunkedPrefill | All | 系统级组合优化 |
| FlashDecoding | 2023 | KV split 并行 | All | Decode: ~3-8x |
| FlashDecoding++ | 2024 | Unified max, 单 kernel | All | ~10-20% vs FD |

## QwenNext 网络架构

| 组件 | 说明 | 数学 |
|------|------|-----|
| RMSNorm | $y = x / \text{RMS}(x) \cdot \gamma$ | $\text{RMS}(x) = \sqrt{\frac{1}{d}\sum x_i^2}$ |
| RoPE | 旋转位置编码, 内积仅依赖相对位置 | $q' = q \cos(m\theta) + \text{rotate\_half}(q) \sin(m\theta)$ |
| GQA | 14 Q heads, 2 KV heads, group=7 | KV cache 减少 86% |
| SwiGLU FFN | $\text{SwiGLU}(x) = W_{down}(\text{silu}(W_{gate}x) \odot W_{up}x)$ | 3 个投影矩阵 |
| KV Cache | Prefill 构建, Decode 递增 | 增量更新 O(1) per token |

**Qwen2-0.5B 规模**: 24 layers, hidden=896, vocab=151936, **~494M params**

## Qwen3-Next 架构升级

| 升级 | 说明 | 核心收益 |
|------|------|----------|
| Hybrid Attention | 3:1 DeltaNet:Softmax 混合 | KV Cache 减少 ~75%, DeltaNet 层 O(1) 推理 |
| Gated DeltaNet | $S_t = \text{diag}(\alpha) S_{t-1} + \beta (v - \beta S^\top k) \otimes k$ | 线性 attention + delta rule, 固定 O(d²) state |
| Gated Attention | $O = \sigma(W_g x) \odot \text{softmax}(QK^T/\sqrt{d})V$ | gate→0 消除 Attention Sink |
| ZC-RMSNorm | $y = x / \text{RMS}(x) \cdot (1+\gamma)$, $\gamma$ 初始化为 0 | 初始 residual 直通, 训练更稳定 |
| MTP | 主 head + k 个辅助 head 预测 t+1, t+2, ... | 投机采样 draft model, 1.5-3x 推理加速 |
| High-Sparsity MoE | 128 experts, top-8 active (6.25%), +1 shared | 参数量大但激活少, 高容量低计算 |

### 推理成本对比 (32K context)

| 指标 | Qwen2 (全 Softmax) | Qwen3 (Hybrid 3:1) |
|------|--------------------|--------------------|  
| Total KV Cache (48L) | 768 MB | ~193 MB (↓75%) |
| Decode FLOPs/token | O(N×d) all layers | O(N×d) 25% + O(d²) 75% |
| Attention Sink | 存在 | Gate 消除 |
| 投机采样 | 需外部 draft | MTP 内置 (~2x) |

## 参考文献

1. **Online Softmax**: Milakov & Gimelshein, "Online normalizer calculation for softmax", 2018
2. **FlashAttention V1**: Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness", NeurIPS 2022
3. **FlashAttention V2**: Dao, "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning", ICLR 2024
4. **FlashAttention V3**: Shah et al., "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision", 2024
5. **FlashDecoding**: Dao et al., "Flash-Decoding for long-context inference", 2023
6. **FlashDecoding++**: Hong et al., "FlashDecoding++: Faster Large Language Model Inference on GPUs", 2024
7. **PagedAttention**: Kwon et al., "Efficient Memory Management for LLM Serving with PagedAttention" (vLLM), SOSP 2023
8. **Ring Attention**: Liu et al., "Ring Attention with Blockwise Transformers for Near-Infinite Context", ICLR 2024
9. **GQA**: Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints", EMNLP 2023
10. **RoPE**: Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding", 2021
11. **SwiGLU**: Shazeer, "GLU Variants Improve Transformer", 2020
12. **RMSNorm**: Zhang & Sennrich, "Root Mean Square Layer Normalization", NeurIPS 2019
13. **Qwen2**: Yang et al., "Qwen2 Technical Report", 2024
14. **Qwen3**: Qwen Team, "Qwen3 Technical Report", 2025
15. **Gated DeltaNet**: Yang et al., "Gated Delta Networks: Improving Mamba2 with Delta Rule", 2024
16. **DeltaNet**: Schlag et al., "Linear Transformers Are Secretly Fast Weight Programmers", ICML 2021
17. **GLA**: Yang et al., "Gated Linear Attention Transformers with Hardware-Efficient Training", ICML 2024
18. **MTP**: DeepSeek-AI, "DeepSeek-V3 Technical Report", 2024
19. **MoE**: Fedus et al., "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity", JMLR 2022
