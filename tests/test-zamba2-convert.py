#!/usr/bin/env python3
# Copyright 2026 bong-water-water-bong
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Zamba2 shared blocks: converter output against the loader alias rule (engine #1036).
# The converter writes each shared block once, at the first num_mem_blocks hybrid layers, and one gate_up adapter pair per hybrid layer.
# The loader (src/models/zamba2.cpp) takes the hybrid layers with attn_output as sources and maps visit v to sources[v % n_sources].
# test-llama-archs makes all tensors itself, so it does not check this map. A wrong map still loads and runs, with the wrong weights.
#
# This test makes tiny random Zamba2 checkpoints with transformers, converts them, applies the loader rule to the GGUF,
# and compares the weights of each hybrid layer with the module that transformers runs at that layer.
# Controls: two swapped blocks must fail the compare; with --bin, a block not at the first hybrid layer must be refused by the loader.
#
#   python3 tests/test-zamba2-convert.py [--bin build/bin] [--keep DIR]
#
# Needs torch, transformers (with Zamba2), safetensors, tokenizers, numpy. Exit 0 pass, 1 fail, 77 skip.

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "gguf-py"))

try:
    import numpy as np
    import torch
    from safetensors.torch import save_file
    from tokenizers import Tokenizer, decoders, models, normalizers
    from transformers import LlamaTokenizerFast, Zamba2Config, Zamba2ForCausalLM
    import gguf
except ImportError as e:  # pragma: no cover - environment check
    print(f"SKIP: {e}")
    sys.exit(77)

# name, num_mem_blocks, attention adapters, hybrid layer ids (of n_layer), n_layer
CASES = [
    ("blocks1",          1, False, [1, 3, 5, 7], 9),
    ("blocks1-attn",     1, True,  [1, 3, 5, 7], 9),   # Zamba2-1.2B-Instruct-v2 shape
    ("blocks2",          2, False, [1, 3, 5, 7, 8], 10),  # 2.7B / 7B-Instruct shape
    ("blocks3",          3, False, [0, 2, 4, 5, 7, 9, 10], 11),
]

RANK = 4
HIDDEN = 32
FF = 48
HEADS = 4


def make_tokenizer(out: Path) -> int:
    vocab = {"<unk>": 0, "<s>": 1, "</s>": 2}
    for b in range(256):
        vocab[f"<0x{b:02X}>"] = len(vocab)
    for t in ["\u2581", "\u2581a", "\u2581b", "a", "b", "c"]:
        vocab[t] = len(vocab)
    tok = Tokenizer(models.BPE(vocab=vocab, merges=[("\u2581", "a"), ("\u2581", "b")], unk_token="<unk>",
                               byte_fallback=True, fuse_unk=True))
    tok.normalizer = normalizers.Sequence([normalizers.Prepend("\u2581"), normalizers.Replace(" ", "\u2581")])
    tok.decoder = decoders.Sequence([decoders.Replace("\u2581", " "), decoders.ByteFallback(), decoders.Fuse(),
                                     decoders.Strip(" ", 1, 0)])
    LlamaTokenizerFast(tokenizer_object=tok, bos_token="<s>", eos_token="</s>", unk_token="<unk>",
                       add_bos_token=True).save_pretrained(out)
    return len(vocab)


def make_checkpoint(out: Path, n_blocks: int, attn_adapter: bool, hybrid: list[int], n_layer: int, seed: int):
    out.mkdir(parents=True)
    n_vocab = make_tokenizer(out)
    cfg = Zamba2Config(
        vocab_size=n_vocab, hidden_size=HIDDEN, intermediate_size=FF, num_hidden_layers=n_layer,
        num_attention_heads=HEADS, num_key_value_heads=HEADS, attention_head_dim=2 * HIDDEN // HEADS,
        layers_block_type=["hybrid" if i in hybrid else "mamba" for i in range(n_layer)],
        num_mem_blocks=n_blocks, use_shared_attention_adapter=attn_adapter, use_shared_mlp_adapter=True,
        adapter_rank=RANK, mamba_d_state=16, mamba_d_conv=4, mamba_expand=2, mamba_headdim=16, mamba_ngroups=1,
        use_mem_rope=True, max_position_embeddings=64, tie_word_embeddings=True,
    )
    torch.manual_seed(seed)
    model = Zamba2ForCausalLM(cfg).eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            # a norm left at 1.0 would compare equal to any other norm
            p.copy_(torch.randn_like(p) * 0.2 + (1.0 if p.ndim == 1 else 0.0))
    # like the published checkpoints: a block is stored only under its first hybrid layer
    first = hybrid[:n_blocks]
    sd = {}
    for k, v in model.state_dict().items():
        m = re.match(r"model\.layers\.(\d+)\.shared_transformer\.", k)
        if m and int(m.group(1)) not in first:
            continue
        if k == "lm_head.weight":
            continue
        sd[k] = v.detach().clone().contiguous()
    save_file(sd, str(out / "model.safetensors"), metadata={"format": "pt"})
    cfg.architectures = ["Zamba2ForCausalLM"]
    cfg.save_pretrained(out)
    return model


def hf_effective(model, hybrid: list[int]) -> dict[int, dict[str, np.ndarray]]:
    """Weights that transformers uses at each hybrid layer."""
    out = {}
    for v, il in enumerate(hybrid):
        st = model.model.layers[il].shared_transformer
        att, ff = st.self_attn, st.feed_forward
        w = {}

        def adapt(base, adapters):
            ad = adapters[v]
            if isinstance(ad, torch.nn.Identity):
                return base
            return base + ad[1].weight.float() @ ad[0].weight.float()

        q, k, vv = att.q_proj.weight.float(), att.k_proj.weight.float(), att.v_proj.weight.float()
        if model.config.use_shared_attention_adapter:
            q = adapt(q, att.linear_q_adapter_list)
            k = adapt(k, att.linear_k_adapter_list)
            vv = adapt(vv, att.linear_v_adapter_list)
        w["attn_q"], w["attn_k"], w["attn_v"] = q, k, vv
        w["attn_output"] = att.o_proj.weight.float()
        w["post_attention_norm"] = st.input_layernorm.weight.float()
        w["ffn_norm"] = st.pre_ff_layernorm.weight.float()
        w["ffn_down"] = ff.down_proj.weight.float()
        w["gate_up"] = adapt(ff.gate_up_proj.weight.float(), ff.gate_up_proj_adapter_list)
        w["_block"] = st.block_id
        out[il] = {kk: (vv_.detach().numpy() if isinstance(vv_, torch.Tensor) else vv_) for kk, vv_ in w.items()}
    return out


SHARED = ["post_attention_norm", "attn_q", "attn_k", "attn_v", "attn_output", "ffn_norm", "ffn_down"]


def gguf_effective(path: Path) -> tuple[dict, dict]:
    """Weights that the zamba2 loader resolves at each hybrid layer."""
    r = gguf.GGUFReader(str(path))
    t = {x.name: np.array(x.data, dtype=np.float32) for x in r.tensors}
    kv = {f.name: f for f in r.fields.values()}
    n_kv = kv["zamba2.attention.head_count_kv"]
    head_kv = [int(n_kv.parts[i][0]) for i in n_kv.data]
    rank_f = kv.get("zamba2.ffn_gate_up_adapter_rank")
    rank = int(rank_f.parts[rank_f.data[0]][0]) if rank_f else 0

    use = [i for i, h in enumerate(head_kv) if h > 0]
    src = [i for i in use if f"blk.{i}.attn_output.weight" in t]
    layers = {}
    for i in use:
        layers[i] = {s: t.get(f"blk.{i}.{s}.weight") for s in SHARED + ["ffn_up"]}
    # the loader's two asserts
    if not src:
        raise AssertionError("loader would refuse: no shared block stored")
    if src != use[:len(src)]:
        raise AssertionError(f"loader would refuse: shared block at {src}, not at the first hybrid layers {use}")
    for v, i in enumerate(use):
        lay = layers[i]
        if lay["attn_output"] is not None:
            continue
        s = layers[src[v % len(src)]]
        own_qkv = lay["attn_q"] is not None  # attention adapters: q/k/v are stored per layer
        for name in SHARED:
            if name in ("attn_q", "attn_k", "attn_v") and own_qkv:
                continue
            lay[name] = s[name]
        if lay["ffn_up"] is None:
            lay["ffn_up"] = s["ffn_up"]
    eff = {}
    for i in use:
        lay = layers[i]
        missing = [n for n in SHARED + ["ffn_up"] if lay[n] is None]
        if missing:
            eff[i] = {"_missing": missing}
            continue
        e = {n: lay[n] for n in SHARED}
        gu = lay["ffn_up"]
        if rank > 0:
            a, b = t[f"blk.{i}.ffn_gate_up_adapter_a.weight"], t[f"blk.{i}.ffn_gate_up_adapter_b.weight"]
            gu = gu + b @ a
        e["gate_up"] = gu
        eff[i] = e
    counts = {
        "ffn_up": sum(1 for n in t if re.fullmatch(r"blk\.\d+\.ffn_up\.weight", n)),
        "attn_q": sum(1 for n in t if re.fullmatch(r"blk\.\d+\.attn_q\.weight", n)),
        "attn_output": sum(1 for n in t if re.fullmatch(r"blk\.\d+\.attn_output\.weight", n)),
        "adapter_a": sum(1 for n in t if n.endswith("ffn_gate_up_adapter_a.weight")),
        "adapter_b": sum(1 for n in t if n.endswith("ffn_gate_up_adapter_b.weight")),
        "rank": rank, "src": src, "n_tensors": len(t),
    }
    return eff, counts


def compare(hf: dict, gg: dict) -> list[str]:
    bad = []
    for il, h in hf.items():
        g = gg.get(il)
        if g is None:
            bad.append(f"blk.{il}: not a hybrid layer in the GGUF")
            continue
        if "_missing" in g:
            bad.append(f"blk.{il} (block {h['_block']}): the loader resolves no {', '.join(g['_missing'])}")
            continue
        for name in SHARED + ["gate_up"]:
            hv, gv = h[name], g[name]
            if hv.shape != gv.shape:
                bad.append(f"blk.{il}.{name}: shape {gv.shape}, transformers runs {hv.shape}")
            elif not np.allclose(hv, gv, rtol=0, atol=1e-5):
                bad.append(f"blk.{il}.{name} (block {h['_block']}): max |diff| {np.abs(hv - gv).max():.3g}")
    return bad


def convert(ckpt: Path, out: Path):
    subprocess.run([sys.executable, str(ROOT / "convert_hf_to_gguf.py"), str(ckpt), "--outfile", str(out),
                    "--outtype", "f32"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)


def rewrite(src: Path, dst: Path, rename):
    """Copy a GGUF and rename its tensors."""
    r = gguf.GGUFReader(str(src))
    w = gguf.GGUFWriter(str(dst), arch="zamba2")
    for f in r.fields.values():
        if f.name.startswith("GGUF.") or f.name == "general.architecture":
            continue
        vt = f.types[0]
        if vt == gguf.GGUFValueType.ARRAY:
            sub = f.types[-1]
            if sub == gguf.GGUFValueType.STRING:
                val = [bytes(f.parts[i]).decode("utf-8") for i in f.data]
            else:
                val = [f.parts[i][0].item() for i in f.data]
            w.add_key_value(f.name, val, vt, sub_type=sub)
        elif vt == gguf.GGUFValueType.STRING:
            w.add_key_value(f.name, bytes(f.parts[f.data[0]]).decode("utf-8"), vt)
        else:
            w.add_key_value(f.name, f.parts[f.data[0]][0].item(), vt)
    for x in r.tensors:
        w.add_tensor(rename(x.name), np.array(x.data))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def run_bin(bindir: Path, model: Path) -> subprocess.CompletedProcess:
    # llama-perplexity runs the full graph and is in every build
    text = model.with_suffix(".txt")
    text.write_text("the quick brown fox jumps over the lazy dog. " * 40)
    exe = bindir / "llama-perplexity"
    return subprocess.run([str(exe), "-m", str(model), "-f", str(text), "-c", "32", "-b", "32", "--chunks", "1",
                           "-t", "2", "-dev", "none", "-ngl", "0"],
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=300)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", type=Path, help="llama.cpp bin dir: also load and evaluate the files, and the loader control")
    ap.add_argument("--keep", type=Path, help="keep the checkpoints and GGUFs here")
    args = ap.parse_args()

    work = args.keep or Path(tempfile.mkdtemp(prefix="zamba2-convert-"))
    work.mkdir(parents=True, exist_ok=True)
    failed = 0
    for seed, (name, n_blocks, attn_ad, hybrid, n_layer) in enumerate(CASES):
        ckpt, out = work / name, work / f"{name}.gguf"
        model = make_checkpoint(ckpt, n_blocks, attn_ad, hybrid, n_layer, seed)
        try:
            convert(ckpt, out)
        except subprocess.CalledProcessError as e:
            print(f"FAIL {name}: convert_hf_to_gguf.py exited {e.returncode}\n{e.stderr[-2000:]}")
            failed += 1
            continue
        hf = hf_effective(model, hybrid)
        try:
            gg, c = gguf_effective(out)
        except AssertionError as e:
            print(f"FAIL {name}: {e}")
            failed += 1
            continue
        want_q = len(hybrid) if attn_ad else n_blocks
        layout = (c["ffn_up"] == n_blocks and c["attn_output"] == n_blocks and c["attn_q"] == want_q
                  and c["adapter_a"] == len(hybrid) and c["adapter_b"] == len(hybrid) and c["rank"] == RANK)
        bad = compare(hf, gg)
        blocks = [hf[i]["_block"] for i in hybrid]
        print(f"{'ok  ' if layout and not bad else 'FAIL'} {name}: blocks per visit {blocks}, sources {c['src']}, "
              f"ffn_up x{c['ffn_up']} attn_q x{c['attn_q']} attn_output x{c['attn_output']} "
              f"adapters {c['adapter_a']}+{c['adapter_b']} rank {c['rank']}, {c['n_tensors']} tensors")
        if not layout:
            print(f"     layout: want ffn_up x{n_blocks}, attn_q x{want_q}, attn_output x{n_blocks}, "
                  f"adapters {len(hybrid)}+{len(hybrid)}, rank {RANK}")
        for b in bad[:12]:
            print(f"     {b}")
        failed += (not layout) or bool(bad)

        if args.bin:
            p = run_bin(args.bin, out)
            ok = p.returncode == 0
            print(f"{'ok  ' if ok else 'FAIL'} {name}: loads and evaluates on the CPU (exit {p.returncode})")
            if not ok:
                print("     " + "\n     ".join(p.stdout.strip().splitlines()[-6:]))
                failed += 1

        # control 1: swap two blocks. The loader accepts this file, so the compare must fail.
        if n_blocks >= 2 and not attn_ad:
            a, b = hybrid[0], hybrid[1]
            sw = work / f"{name}-swapped.gguf"

            def swap(n, a=a, b=b):
                m = re.fullmatch(r"blk\.(\d+)\.(post_attention_norm|attn_[qkv]|attn_output|ffn_norm|ffn_up|ffn_down)\.weight", n)
                if not m or int(m.group(1)) not in (a, b):
                    return n
                return f"blk.{b if int(m.group(1)) == a else a}.{m.group(2)}.weight"
            rewrite(out, sw, swap)
            nbad = len(compare(hf, gguf_effective(sw)[0]))
            ok = nbad > 0
            print(f"{'ok  ' if ok else 'FAIL'} {name}: control, blocks swapped at blk.{a}/blk.{b} -> {nbad} mismatches")
            failed += not ok

        # control 2: the block is at the second hybrid layer. The loader must refuse this file.
        if n_blocks == 1 and not attn_ad:
            a, b = hybrid[0], hybrid[1]
            mv = work / f"{name}-moved.gguf"

            def move(n, a=a, b=b):
                m = re.fullmatch(r"blk\.(\d+)\.(post_attention_norm|attn_[qkv]|attn_output|ffn_norm|ffn_up|ffn_down)\.weight", n)
                return f"blk.{b}.{m.group(2)}.weight" if m and int(m.group(1)) == a else n
            rewrite(out, mv, move)
            try:
                gguf_effective(mv)
                print(f"FAIL {name}: control, block moved to blk.{b}: this test's loader model accepted it")
                failed += 1
            except AssertionError:
                print(f"ok   {name}: control, block moved to blk.{b}: refused by the loader model")
            if args.bin:
                p = run_bin(args.bin, mv)
                refused = p.returncode != 0 and "first hybrid layers" in p.stdout
                print(f"{'ok  ' if refused else 'FAIL'} {name}: control, block moved to blk.{b}: "
                      f"llama.cpp exit {p.returncode}{', refused by the loader assert' if refused else ''}")
                failed += not refused

    if not args.keep:
        import shutil
        shutil.rmtree(work, ignore_errors=True)
    print("PASS" if not failed else f"FAILED: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
