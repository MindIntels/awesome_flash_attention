#!/usr/bin/env python3
"""
FlashAttention 全链路测试脚本

运行方式:
    python run_all_tests.py              # 运行所有测试
    python run_all_tests.py --quick      # 快速测试 (小规模)
    python run_all_tests.py --benchmark  # 包含性能基准测试
    python run_all_tests.py --step 3     # 只运行指定步骤
"""

import argparse
import importlib
import sys
import os
import time
import torch
import torch.nn.functional as F
import math
from typing import Optional

# 添加当前目录到 path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def reference_attention(Q, K, V, causal=False):
    """标准 attention reference 实现"""
    scale = 1.0 / math.sqrt(Q.shape[-1])
    S = torch.matmul(Q, K.transpose(-2, -1)) * scale
    if causal:
        N = S.shape[-1]
        mask = torch.triu(torch.ones(N, N, dtype=torch.bool, device=S.device), diagonal=1)
        S.masked_fill_(mask, float('-inf'))
    P = F.softmax(S, dim=-1)
    return torch.matmul(P, V)


class TestRunner:
    def __init__(self, quick=False, verbose=True):
        self.quick = quick
        self.verbose = verbose
        self.results = []
        self.passed = 0
        self.failed = 0

    def check(self, name: str, O_test: torch.Tensor, O_ref: torch.Tensor,
              atol: float = 1e-4):
        diff = (O_test - O_ref).abs().max().item()
        ok = diff < atol
        if ok:
            self.passed += 1
        else:
            self.failed += 1
        status = "PASS" if ok else "FAIL"
        self.results.append((name, status, diff))
        if self.verbose:
            color = "\033[92m" if ok else "\033[91m"
            reset = "\033[0m"
            print(f"  {color}[{status}]{reset} {name:50s}  max_diff = {diff:.2e}")
        return ok

    def summary(self):
        print("\n" + "=" * 70)
        total = self.passed + self.failed
        color = "\033[92m" if self.failed == 0 else "\033[91m"
        reset = "\033[0m"
        print(f"{color}测试结果: {self.passed}/{total} 通过, {self.failed} 失败{reset}")
        if self.failed > 0:
            print("失败项:")
            for name, status, diff in self.results:
                if status == "FAIL":
                    print(f"  - {name} (diff={diff:.2e})")
        print("=" * 70)
        return self.failed == 0


def test_step1_softmax(runner: TestRunner):
    """测试 01: Softmax 演化"""
    print("\n" + "=" * 70)
    print("Step 1: Softmax Evolution")
    print("=" * 70)

    from importlib import import_module
    mod = import_module("01_softmax_evolution")

    sizes = [64, 256] if runner.quick else [64, 256, 1024, 4096]

    for N in sizes:
        x = torch.randn(N)
        ref = F.softmax(x, dim=-1)

        # Naive softmax
        naive = mod.naive_softmax(x)
        runner.check(f"naive_softmax       N={N}", naive, ref, atol=1e-5)

        # Safe softmax (3-pass)
        safe = mod.safe_softmax_3pass(x)
        runner.check(f"safe_softmax_3pass  N={N}", safe, ref, atol=1e-6)

        # Online softmax (2-pass)
        online = mod.online_softmax_2pass(x)
        runner.check(f"online_softmax_2pass N={N}", online, ref, atol=1e-5)


def test_step2_memory_efficient(runner: TestRunner):
    """测试 02: Memory Efficient Attention"""
    print("\n" + "=" * 70)
    print("Step 2: Memory Efficient Attention")
    print("=" * 70)

    mod = importlib.import_module("02_memory_efficient_attention")

    B, H, d = 2, 4, 64
    sizes = [64, 128] if runner.quick else [64, 128, 256, 512]

    for N in sizes:
        Q = torch.randn(B, H, N, d)
        K = torch.randn(B, H, N, d)
        V = torch.randn(B, H, N, d)
        O_ref = reference_attention(Q, K, V)

        O_std_result = mod.StandardAttention.forward(Q, K, V)
        O_std = O_std_result[0] if isinstance(O_std_result, tuple) else O_std_result
        runner.check(f"StandardAttention    N={N}", O_std, O_ref, atol=1e-5)

        O_mem = mod.MemoryEfficientAttention.forward(Q, K, V)
        O_mem = O_mem[0] if isinstance(O_mem, tuple) else O_mem
        runner.check(f"MemoryEfficientAttn  N={N}", O_mem, O_ref, atol=1e-4)


def test_step3_flash_v1(runner: TestRunner):
    """测试 03: FlashAttention V1"""
    print("\n" + "=" * 70)
    print("Step 3: FlashAttention V1")
    print("=" * 70)

    mod = importlib.import_module("03_flash_attention_v1")

    B, H, d = 2, 4, 64
    sizes = [64, 128] if runner.quick else [64, 128, 256]

    for N in sizes:
        Q = torch.randn(B, H, N, d)
        K = torch.randn(B, H, N, d)
        V = torch.randn(B, H, N, d)
        O_ref = reference_attention(Q, K, V)

        O_v1_result = mod.FlashAttentionV1.forward(Q, K, V, block_size_r=32, block_size_c=32)
        O_v1 = O_v1_result[0] if isinstance(O_v1_result, tuple) else O_v1_result
        runner.check(f"FlashAttnV1          N={N}", O_v1, O_ref, atol=1e-4)

        # Causal
        O_ref_c = reference_attention(Q, K, V, causal=True)
        O_v1_c_result = mod.FlashAttentionV1.forward(Q, K, V, block_size_r=32, block_size_c=32, causal=True)
        O_v1_c = O_v1_c_result[0] if isinstance(O_v1_c_result, tuple) else O_v1_c_result
        runner.check(f"FlashAttnV1 causal   N={N}", O_v1_c, O_ref_c, atol=1e-4)


def test_step4_flash_v2(runner: TestRunner):
    """测试 04: FlashAttention V2"""
    print("\n" + "=" * 70)
    print("Step 4: FlashAttention V2")
    print("=" * 70)

    mod = importlib.import_module("04_flash_attention_v2")

    B, H, d = 2, 4, 64
    sizes = [64, 128] if runner.quick else [64, 128, 256]

    for N in sizes:
        Q = torch.randn(B, H, N, d)
        K = torch.randn(B, H, N, d)
        V = torch.randn(B, H, N, d)
        O_ref = reference_attention(Q, K, V)

        O_v2_result = mod.FlashAttentionV2.forward(Q, K, V, block_size_q=32, block_size_kv=32)
        O_v2 = O_v2_result[0] if isinstance(O_v2_result, tuple) else O_v2_result
        runner.check(f"FlashAttnV2          N={N}", O_v2, O_ref, atol=1e-4)

        # Causal
        O_ref_c = reference_attention(Q, K, V, causal=True)
        O_v2_c_result = mod.FlashAttentionV2.forward(Q, K, V, block_size_q=32, block_size_kv=32, causal=True)
        O_v2_c = O_v2_c_result[0] if isinstance(O_v2_c_result, tuple) else O_v2_c_result
        runner.check(f"FlashAttnV2 causal   N={N}", O_v2_c, O_ref_c, atol=1e-4)


def test_step5_flash_decoding(runner: TestRunner):
    """测试 05: FlashDecoding & FlashDecoding++"""
    print("\n" + "=" * 70)
    print("Step 5: FlashDecoding & FlashDecoding++")
    print("=" * 70)

    mod = importlib.import_module("05_flash_decoding")

    B, H, d = 2, 8, 64
    kv_sizes = [256, 512] if runner.quick else [256, 512, 1024, 2048]

    for N_k in kv_sizes:
        Q = torch.randn(B, H, 1, d)
        K = torch.randn(B, H, N_k, d)
        V = torch.randn(B, H, N_k, d)
        O_ref = reference_attention(Q, K, V)

        # FlashDecoding
        for ns in [4, 8]:
            O_fd = mod.FlashDecoding.forward(Q, K, V, num_splits=ns)
            runner.check(f"FlashDecoding    N_k={N_k} s={ns}", O_fd, O_ref, atol=1e-4)

        # FlashDecoding++
        O_fdpp = mod.FlashDecodingPP.forward(Q, K, V, num_splits=4)
        runner.check(f"FlashDecoding++  N_k={N_k}", O_fdpp, O_ref, atol=1e-4)


def test_step6_flash_v3(runner: TestRunner):
    """测试 06: FlashAttention V3"""
    print("\n" + "=" * 70)
    print("Step 6: FlashAttention V3 (Simulation)")
    print("=" * 70)

    mod = importlib.import_module("06_flash_attention_v3")

    B, H, d = 2, 4, 64
    sizes = [64, 128] if runner.quick else [64, 128, 256]

    for N in sizes:
        Q = torch.randn(B, H, N, d)
        K = torch.randn(B, H, N, d)
        V = torch.randn(B, H, N, d)
        O_ref = reference_attention(Q, K, V)

        # Warp specialized
        O_ws, stats = mod.FlashAttentionV3Sim.forward_warp_specialized(Q, K, V, B_r=64, B_c=64)
        runner.check(f"FlashAttnV3 warp-spec N={N}", O_ws, O_ref, atol=1e-4)

        # Ping-pong
        O_pp = mod.FlashAttentionV3Sim.forward_pingpong(Q, K, V, B_r=64, B_c=64)
        runner.check(f"FlashAttnV3 ping-pong N={N}", O_pp, O_ref, atol=1e-4)

        # Causal
        O_ref_c = reference_attention(Q, K, V, causal=True)
        O_ws_c, _ = mod.FlashAttentionV3Sim.forward_warp_specialized(Q, K, V, B_r=64, B_c=64, causal=True)
        runner.check(f"FlashAttnV3 causal    N={N}", O_ws_c, O_ref_c, atol=1e-4)

    # FP8 (looser tolerance)
    N = 128
    Q = torch.randn(B, H, N, d)
    K = torch.randn(B, H, N, d)
    V = torch.randn(B, H, N, d)
    O_ref = reference_attention(Q, K, V)
    O_fp8, _ = mod.FlashAttentionV3Sim.forward_warp_specialized(Q, K, V, simulate_fp8=True)
    runner.check(f"FlashAttnV3 FP8 sim   N={N}", O_fp8, O_ref, atol=0.5)


def test_step7_flash_v4(runner: TestRunner):
    """测试 07: FlashAttention V4 (PagedKV, Ring, GQA, SlidingWindow, ChunkedPrefill)"""
    print("\n" + "=" * 70)
    print("Step 7: FlashAttention V4")
    print("=" * 70)

    mod = importlib.import_module("07_flash_attention_v4")
    torch.manual_seed(42)
    B, H, d = 1, 4, 64

    # ── Ring Attention ──
    for N in ([64, 128] if runner.quick else [64, 128, 256]):
        Q = torch.randn(B, H, N, d)
        K = torch.randn(B, H, N, d)
        V = torch.randn(B, H, N, d)
        O_ref = reference_attention(Q, K, V)
        for P in [2, 4]:
            if P > N:
                continue
            O_ring = mod.RingAttention.forward(Q, K, V, num_devices=P)
            runner.check(f"RingAttention   N={N} P={P}", O_ring, O_ref, atol=1e-4)

    # ── GQA FlashAttention ──
    B_t = 2
    for H_q, H_kv, name in [(8, 2, "GQA8/2"), (8, 1, "MQA8/1")]:
        N = 64
        Q = torch.randn(B_t, H_q, N, d)
        K = torch.randn(B_t, H_kv, N, d)
        V = torch.randn(B_t, H_kv, N, d)
        O_gqa = mod.GQAFlashAttention.forward(Q, K, V, num_kv_heads=H_kv, block_size=32)
        group_size = H_q // H_kv
        K_exp = K.unsqueeze(2).expand(-1, -1, group_size, -1, -1).reshape(B_t, H_q, N, d)
        V_exp = V.unsqueeze(2).expand(-1, -1, group_size, -1, -1).reshape(B_t, H_q, N, d)
        O_ref = reference_attention(Q, K_exp, V_exp)
        runner.check(f"GQA FlashAttn {name}  N={N}", O_gqa, O_ref, atol=1e-4)

    # ── Sliding Window ──
    for N, W in [(64, 32), (128, 64)]:
        Q = torch.randn(1, H, N, d)
        K = torch.randn(1, H, N, d)
        V = torch.randn(1, H, N, d)
        scale = 1.0 / math.sqrt(d)
        S = torch.matmul(Q, K.transpose(-2, -1)) * scale
        row_idx = torch.arange(N).unsqueeze(1)
        col_idx = torch.arange(N).unsqueeze(0)
        mask = (row_idx < col_idx) | ((row_idx - col_idx) >= W)
        S.masked_fill_(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        O_ref = torch.matmul(F.softmax(S, dim=-1), V)
        O_sw = mod.SlidingWindowFlashAttention.forward(Q, K, V, window_size=W, block_size=32, causal=True)
        runner.check(f"SlidingWindow   N={N} W={W}", O_sw, O_ref, atol=1e-4)

    # ── Paged KV Cache ──
    cache = mod.PagedKVCache(num_heads=4, head_dim=64, page_size=16, max_pages=64)
    cache.allocate(seq_id=0, num_tokens=32)
    k_data = torch.randn(4, 32, 64)
    v_data = torch.randn(4, 32, 64)
    cache.append(seq_id=0, k=k_data, v=v_data)
    K_read, V_read = cache.read(seq_id=0)
    diff = (K_read - k_data).abs().max().item()
    ok = diff < 1e-6
    runner.results.append(("PagedKVCache read/write", "PASS" if ok else "FAIL", diff))
    if ok:
        runner.passed += 1
    else:
        runner.failed += 1
    if runner.verbose:
        color = "\033[92m" if ok else "\033[91m"
        print(f"  {color}[{'PASS' if ok else 'FAIL'}]\033[0m PagedKVCache read/write                          max_diff = {diff:.2e}")
    cache.free(seq_id=0)


def test_step8_qwen_next(runner: TestRunner):
    """测试 08: QwenNext 完整网络"""
    print("\n" + "=" * 70)
    print("Step 8: QwenNext (GQA + RoPE + SwiGLU + FlashAttention)")
    print("=" * 70)

    mod = importlib.import_module("08_qwen_next")
    torch.manual_seed(42)

    config = mod.QwenNextConfig(
        vocab_size=1000, hidden_size=128, intermediate_size=256,
        num_layers=2, num_heads=8, num_kv_heads=2, head_dim=16,
        max_seq_len=256,
    )

    # ── RMSNorm ──
    norm = mod.RMSNorm(64)
    x = torch.randn(2, 10, 64)
    y = norm(x)
    rms = torch.sqrt(y.pow(2).mean(-1))
    rms_ok = (rms.mean().item() - 1.0) < 0.1
    runner.results.append(("RMSNorm mean_rms≈1", "PASS" if rms_ok else "FAIL", abs(rms.mean().item() - 1.0)))
    if rms_ok: runner.passed += 1
    else: runner.failed += 1
    if runner.verbose:
        color = "\033[92m" if rms_ok else "\033[91m"
        print(f"  {color}[{'PASS' if rms_ok else 'FAIL'}]\033[0m RMSNorm mean_rms≈1.0                                mean_rms = {rms.mean().item():.4f}")

    # ── RoPE 范数保持 ──
    rope = mod.RotaryPositionEmbedding(head_dim=64, max_seq_len=128)
    q = torch.randn(2, 4, 16, 64)
    k = torch.randn(2, 4, 16, 64)
    pos = torch.arange(16).unsqueeze(0).expand(2, -1)
    q_rot, k_rot = rope(q, k, pos)
    norm_diff = (q.norm(dim=-1) - q_rot.norm(dim=-1)).abs().max().item()
    runner.check("RoPE norm preservation", q.norm(dim=-1), q_rot.norm(dim=-1), atol=1e-4)

    # ── Full model prefill + decode ──
    model = mod.QwenNextModel(config)
    B, N = 2, 16
    input_ids = torch.randint(0, config.vocab_size, (B, N))

    logits, kv_caches = model(input_ids)
    prefill_ok = logits.shape == (B, N, config.vocab_size)
    runner.results.append(("QwenNext Prefill shape", "PASS" if prefill_ok else "FAIL", 0.0))
    if prefill_ok: runner.passed += 1
    else: runner.failed += 1
    if runner.verbose:
        color = "\033[92m" if prefill_ok else "\033[91m"
        print(f"  {color}[{'PASS' if prefill_ok else 'FAIL'}]\033[0m QwenNext Prefill shape                              {logits.shape}")

    # Decode
    next_tok = logits[:, -1, :].argmax(dim=-1)
    logits_d, kv_d = model(next_tok.unsqueeze(1), kv_caches=kv_caches)
    decode_ok = logits_d.shape == (B, 1, config.vocab_size) and kv_d[0][0].shape[2] == N + 1
    runner.results.append(("QwenNext Decode step", "PASS" if decode_ok else "FAIL", 0.0))
    if decode_ok: runner.passed += 1
    else: runner.failed += 1
    if runner.verbose:
        color = "\033[92m" if decode_ok else "\033[91m"
        print(f"  {color}[{'PASS' if decode_ok else 'FAIL'}]\033[0m QwenNext Decode step                                KV_len={kv_d[0][0].shape[2]}")

    # Generate
    gen = mod.QwenNextGenerator(model)
    prompt = torch.randint(0, config.vocab_size, (1, 4))
    output = gen.generate(prompt, max_new_tokens=5, temperature=0.8, top_k=50)
    gen_ok = output.shape[1] == 4 + 5
    runner.results.append(("QwenNext Generate", "PASS" if gen_ok else "FAIL", 0.0))
    if gen_ok: runner.passed += 1
    else: runner.failed += 1
    if runner.verbose:
        color = "\033[92m" if gen_ok else "\033[91m"
        print(f"  {color}[{'PASS' if gen_ok else 'FAIL'}]\033[0m QwenNext Generate                                    output_len={output.shape[1]}")


def test_step9_qwen3_next(runner: TestRunner):
    """测试 09: Qwen3-Next (Hybrid Attention + MoE + MTP)"""
    print("\n" + "=" * 70)
    print("Step 9: Qwen3-Next (Hybrid DeltaNet/Softmax + MoE + MTP)")
    print("=" * 70)

    mod = importlib.import_module("09_qwen3_next")
    torch.manual_seed(42)

    config = mod.Qwen3NextConfig(
        vocab_size=1000, hidden_size=128, intermediate_size=256,
        num_layers=8, num_heads=8, num_kv_heads=2, head_dim=16,
        max_seq_len=256,
        num_experts=8, num_experts_per_tok=2, num_shared_experts=1,
        use_moe=True, use_mtp=True, mtp_num_heads=1,
    )

    # ── ZeroCenteredRMSNorm ──
    norm = mod.ZeroCenteredRMSNorm(64)
    x = torch.randn(2, 10, 64)
    y = norm(x)
    rms_post = torch.sqrt(y.pow(2).mean(-1))
    rms_ok = abs(rms_post.mean().item() - 1.0) < 0.15
    runner.results.append(("ZC-RMSNorm mean_rms≈1", "PASS" if rms_ok else "FAIL",
                           abs(rms_post.mean().item() - 1.0)))
    if rms_ok: runner.passed += 1
    else: runner.failed += 1
    if runner.verbose:
        color = "\033[92m" if rms_ok else "\033[91m"
        print(f"  {color}[{'PASS' if rms_ok else 'FAIL'}]\033[0m ZC-RMSNorm (weight=0 init)                          mean_rms={rms_post.mean().item():.4f}")

    # ── Hybrid layer pattern (3:1) ──
    pattern = "".join("D" if config.is_deltanet_layer(i) else "S" for i in range(config.num_layers))
    pattern_ok = (pattern.count('D') == 6 and pattern.count('S') == 2)  # 8 layers: DDDSDDDS
    runner.results.append(("Hybrid 3:1 pattern", "PASS" if pattern_ok else "FAIL", 0.0))
    if pattern_ok: runner.passed += 1
    else: runner.failed += 1
    if runner.verbose:
        color = "\033[92m" if pattern_ok else "\033[91m"
        print(f"  {color}[{'PASS' if pattern_ok else 'FAIL'}]\033[0m Hybrid layer pattern: {pattern}                       D:S={pattern.count('D')}:{pattern.count('S')}")

    # ── GatedDeltaNet ──
    deltanet = mod.GatedDeltaNet(config)
    dx = torch.randn(2, 8, config.hidden_size)
    dpos = torch.arange(8).unsqueeze(0).expand(2, -1)
    dout, dstate = deltanet(dx, dpos)
    delta_ok = dout.shape == (2, 8, config.hidden_size) and dstate.shape[2] == config.head_dim
    runner.results.append(("GatedDeltaNet forward", "PASS" if delta_ok else "FAIL", 0.0))
    if delta_ok: runner.passed += 1
    else: runner.failed += 1
    if runner.verbose:
        color = "\033[92m" if delta_ok else "\033[91m"
        print(f"  {color}[{'PASS' if delta_ok else 'FAIL'}]\033[0m GatedDeltaNet: state={dstate.shape} (fixed O(d²))   PASS")

    # ── GatedSoftmaxAttention ──
    gattn = mod.GatedSoftmaxAttention(config)
    gx = torch.randn(2, 16, config.hidden_size)
    gpos = torch.arange(16).unsqueeze(0).expand(2, -1)
    gout, gkv = gattn(gx, gpos)
    gattn_ok = gout.shape == (2, 16, config.hidden_size) and gkv[0].shape[2] == 16
    runner.results.append(("GatedSoftmaxAttention", "PASS" if gattn_ok else "FAIL", 0.0))
    if gattn_ok: runner.passed += 1
    else: runner.failed += 1
    if runner.verbose:
        color = "\033[92m" if gattn_ok else "\033[91m"
        print(f"  {color}[{'PASS' if gattn_ok else 'FAIL'}]\033[0m GatedSoftmaxAttention (anti-sink gate)              KV={gkv[0].shape}")

    # ── MoE ──
    moe = mod.HighSparsityMoE(config)
    mx = torch.randn(2, 8, config.hidden_size)
    mout, mloss = moe(mx)
    moe_ok = mout.shape == mx.shape and mloss.item() > 0
    runner.results.append(("HighSparsityMoE forward", "PASS" if moe_ok else "FAIL", 0.0))
    if moe_ok: runner.passed += 1
    else: runner.failed += 1
    if runner.verbose:
        color = "\033[92m" if moe_ok else "\033[91m"
        print(f"  {color}[{'PASS' if moe_ok else 'FAIL'}]\033[0m MoE ({config.num_experts}E/{config.num_experts_per_tok}A): balance_loss={mloss.item():.3f}")

    # ── MTP ──
    mtp = mod.MTPModule(config)
    embed = mod.nn.Embedding(config.vocab_size, config.hidden_size)
    mh = torch.randn(2, 16, config.hidden_size)
    mtids = torch.randint(0, config.vocab_size, (2, 16))
    mtloss, mtlogits = mtp.forward_train(mh, mtids, embed, embed.weight)
    mtp_ok = mtloss.item() > 0 and len(mtlogits) > 0
    runner.results.append(("MTP training loss", "PASS" if mtp_ok else "FAIL", 0.0))
    if mtp_ok: runner.passed += 1
    else: runner.failed += 1
    if runner.verbose:
        color = "\033[92m" if mtp_ok else "\033[91m"
        print(f"  {color}[{'PASS' if mtp_ok else 'FAIL'}]\033[0m MTP training: loss={mtloss.item():.3f}, heads={len(mtlogits)}")

    # ── Full model forward ──
    model = mod.Qwen3NextModel(config)
    B, N = 2, 16
    input_ids = torch.randint(0, config.vocab_size, (B, N))
    target_ids = torch.randint(0, config.vocab_size, (B, N))
    result = model(input_ids, target_ids=target_ids)
    logits = result["logits"]
    fwd_ok = logits.shape == (B, N, config.vocab_size) and "mtp_loss" in result
    runner.results.append(("Qwen3Next full forward", "PASS" if fwd_ok else "FAIL", 0.0))
    if fwd_ok: runner.passed += 1
    else: runner.failed += 1
    if runner.verbose:
        color = "\033[92m" if fwd_ok else "\033[91m"
        print(f"  {color}[{'PASS' if fwd_ok else 'FAIL'}]\033[0m Full model: logits={logits.shape} moe_loss={result['moe_loss'].item():.2f} mtp_loss={result['mtp_loss'].item():.2f}")

    # ── Decode ──
    caches = result["new_caches"]
    next_tok = logits[:, -1, :].argmax(dim=-1)
    res2 = model(next_tok.unsqueeze(1), layer_caches=caches)
    dec_ok = res2["logits"].shape == (B, 1, config.vocab_size)
    runner.results.append(("Qwen3Next decode step", "PASS" if dec_ok else "FAIL", 0.0))
    if dec_ok: runner.passed += 1
    else: runner.failed += 1
    if runner.verbose:
        color = "\033[92m" if dec_ok else "\033[91m"
        print(f"  {color}[{'PASS' if dec_ok else 'FAIL'}]\033[0m Decode: logits={res2['logits'].shape}")

    # ── MTP Speculative Draft ──
    if model.mtp is not None:
        last_h = result["hidden_states"][:, -1:, :]
        last_tok = logits[:, -1, :].argmax(dim=-1)
        drafts = model.mtp.speculative_draft(
            last_h, last_tok, model.embed_tokens, model.get_lm_head_weight())
        draft_ok = len(drafts) == config.mtp_num_heads
        runner.results.append(("MTP speculative draft", "PASS" if draft_ok else "FAIL", 0.0))
        if draft_ok: runner.passed += 1
        else: runner.failed += 1
        if runner.verbose:
            color = "\033[92m" if draft_ok else "\033[91m"
            print(f"  {color}[{'PASS' if draft_ok else 'FAIL'}]\033[0m MTP draft: {len(drafts)} candidate(s)")


def run_benchmark():
    """性能基准测试 (CPU 模拟)"""
    print("\n" + "=" * 70)
    print("性能基准测试 (CPU — PyTorch 模拟, 非 GPU kernel)")
    print("=" * 70)
    print("注意: 这是算法模拟, 不是真实 GPU kernel 的性能测量")

    B, H, d = 1, 4, 64

    methods = {}

    # Import all modules
    mod2 = importlib.import_module("02_memory_efficient_attention")
    mod3 = importlib.import_module("03_flash_attention_v1")
    mod4 = importlib.import_module("04_flash_attention_v2")
    mod5 = importlib.import_module("05_flash_decoding")
    mod6 = importlib.import_module("06_flash_attention_v3")

    def standard_attn(Q, K, V):
        return reference_attention(Q, K, V)

    methods["Standard"] = standard_attn
    methods["MemEfficient"] = lambda Q, K, V: mod2.MemoryEfficientAttention.forward(Q, K, V)
    methods["FlashV1"] = lambda Q, K, V: mod3.FlashAttentionV1.forward(Q, K, V, block_size_r=32, block_size_c=32)[0]
    methods["FlashV2"] = lambda Q, K, V: mod4.FlashAttentionV2.forward(Q, K, V, block_size_q=32, block_size_kv=32)[0]
    methods["FlashV3"] = lambda Q, K, V: mod6.FlashAttentionV3Sim.forward_warp_specialized(Q, K, V, B_r=64, B_c=64)[0]

    sizes = [64, 128, 256, 512]
    print(f"\n{'Method':<15s}", end="")
    for N in sizes:
        print(f"  {'N=' + str(N):>10s}", end="")
    print()
    print("-" * (15 + 12 * len(sizes)))

    for name, fn in methods.items():
        print(f"{name:<15s}", end="")
        for N in sizes:
            torch.manual_seed(0)
            Q = torch.randn(B, H, N, d)
            K = torch.randn(B, H, N, d)
            V = torch.randn(B, H, N, d)

            # Warmup
            fn(Q, K, V)

            # Benchmark
            n_iter = 5
            t0 = time.perf_counter()
            for _ in range(n_iter):
                fn(Q, K, V)
            t1 = time.perf_counter()
            ms = (t1 - t0) / n_iter * 1000
            print(f"  {ms:>8.2f}ms", end="")
        print()

    # Decode benchmark
    print(f"\n[Decode 场景] Q_len=1:")
    print(f"{'Method':<20s}", end="")
    kv_sizes = [512, 1024, 2048]
    for N_k in kv_sizes:
        print(f"  {'KV=' + str(N_k):>10s}", end="")
    print()
    print("-" * (20 + 12 * len(kv_sizes)))

    decode_methods = {
        "Standard": standard_attn,
        "FlashDecoding(s=4)": lambda Q, K, V: mod5.FlashDecoding.forward(Q, K, V, num_splits=4),
        "FlashDecoding(s=8)": lambda Q, K, V: mod5.FlashDecoding.forward(Q, K, V, num_splits=8),
        "FlashDecoding++": lambda Q, K, V: mod5.FlashDecodingPP.forward(Q, K, V, num_splits=4),
    }

    for name, fn in decode_methods.items():
        print(f"{name:<20s}", end="")
        for N_k in kv_sizes:
            Q = torch.randn(1, 8, 1, d)
            K = torch.randn(1, 8, N_k, d)
            V = torch.randn(1, 8, N_k, d)
            fn(Q, K, V)  # warmup
            n_iter = 10
            t0 = time.perf_counter()
            for _ in range(n_iter):
                fn(Q, K, V)
            t1 = time.perf_counter()
            ms = (t1 - t0) / n_iter * 1000
            print(f"  {ms:>8.2f}ms", end="")
        print()


def main():
    parser = argparse.ArgumentParser(description="FlashAttention 全链路测试")
    parser.add_argument("--quick", action="store_true", help="快速测试 (小规模)")
    parser.add_argument("--benchmark", action="store_true", help="运行性能基准测试")
    parser.add_argument("--step", type=int, default=None, help="只跑某一步 (1-8)")
    parser.add_argument("--quiet", action="store_true", help="静默模式")
    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  FlashAttention 全链路测试                                   ║")
    print("║  Softmax → FlashV1/V2/V3/V4 → FlashDecoding → Qwen3-Next   ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    torch.manual_seed(42)
    runner = TestRunner(quick=args.quick, verbose=not args.quiet)

    test_fns = {
        1: ("01_softmax_evolution", test_step1_softmax),
        2: ("02_memory_efficient_attention", test_step2_memory_efficient),
        3: ("03_flash_attention_v1", test_step3_flash_v1),
        4: ("04_flash_attention_v2", test_step4_flash_v2),
        5: ("05_flash_decoding", test_step5_flash_decoding),
        6: ("06_flash_attention_v3", test_step6_flash_v3),
        7: ("07_flash_attention_v4", test_step7_flash_v4),
        8: ("08_qwen_next", test_step8_qwen_next),
        9: ("09_qwen3_next", test_step9_qwen3_next),
    }

    steps = [args.step] if args.step else list(test_fns.keys())

    for step in steps:
        if step not in test_fns:
            print(f"未知步骤: {step}")
            continue
        name, fn = test_fns[step]
        try:
            fn(runner)
        except Exception as e:
            print(f"\033[91m  [ERROR] Step {step} ({name}): {e}\033[0m")
            import traceback
            traceback.print_exc()
            runner.failed += 1

    all_pass = runner.summary()

    if args.benchmark:
        run_benchmark()

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
