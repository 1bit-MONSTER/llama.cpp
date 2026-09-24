# HRX0 Q4NX port — inventory & ABI mapping (task-1)

Repo: `/home/bcloud/hrx-ws/hrx-v2-src` · revision `b08d51080` (detached HEAD)
Date: 2026-09-24
Goal: `mufpkzv5-bqyi90` — *Port the Q4NX dequant + fused matmul kernels from the ggml-hrx2 backend into the HRX0 backend (ggml/src/ggml-hrx) so Q4NX models load and decode correctly on HRX0 with parity to the reference path.*

---

## 1. Problem statement

Q4NX is a **signed 4-bit (int4)** weight format (`GGML_TYPE_Q4NX`, value 42) stored as 5120-byte tiles
of `[32 BF16 rows x 256 cols]`. Q4NX models do not load on the **HRX0** backend
(`ggml/src/ggml-hrx`), because the Q4NX kernels exist only in the **HRX2** backend
(`ggml/src/ggml-hrx2`, Loom DSL), and HRX0 (HIP C++ kernels) has **zero** Q4NX support:

```
$ grep -rn 'Q4NX' ggml/src/ggml-hrx/ | wc -l
0
```

---

## 2. Reproduction — exact failures

Binary: `build-hrx0-latest/bin/llama-cli` (built 2026-09-23). Model tensor types were read from the
GGUF headers directly (280 Q4NX tensors per model):

| Model | Q4NX tensors | type | `blk.0.ffn_down_exps.weight` dims |
|---|---|---|---|
| `store43/zaya1-8b-ft-q4nx.gguf` | 280 | **43** (`Q4NX_C43`) | `[8192, 8192]` |
| `store43/zaya1-8b-ft-q4nx-c42.gguf` | 280 | 42 (`Q4NX`) | `[8192, 8192]` |
| `store43/zaya1-8b-ft-q4nx-repacked.gguf` | 280 | 42 | `[8192, 512, 16]` (3-D, MoE-stacked) |
| `zaya-q4nx-gemma4tok.gguf` | 280 | 42 | `[8192, 512, 16]` |

### Failure A — type 43 is not registered in ggml (model-load, GGUF parse)

```
gguf_init_from_file_ptr: tensor 'blk.0.ffn_down_exps.weight' has invalid ggml type 43. should be in [0, 43)
llama_model_load: error loading model: llama_model_loader: failed to load model from .../zaya1-8b-ft-q4nx.gguf
```

`ggml/include/ggml.h` defines only `GGML_TYPE_Q4NX = 42` and `GGML_TYPE_COUNT = 43`, so the valid
range is `[0,43)` and **43 is undefined**. Branch HEAD's `b08d51080 "gguf-py: register Q4NX_C43 (type 43)"`
registered 43 in the Python gguf tooling only; the C enum was not extended.

### Failure B — no buffer type accepts Q4NX on HRX0 (the real kernel gap)

Loading the type-42 variant gets past GGUF parsing and fails at weight placement:

```
llama_model_load: error loading model: failed to find a compatible buffer type for tensor blk.0.attn_q.weight
```

Mechanism (traced):

1. `src/llama-model-loader.cpp:1196` calls `select_weight_buft(...)`; on `nullptr` it throws at
   `src/llama-model-loader.cpp:1208` ("failed to find a compatible buffer type"). `select_weight_buft`
   only returns buffer types whose **device `supports_op` accepts the tensor's op**.
2. **CPU explicitly rejects Q4NX** — `ggml/src/ggml-cpu/ggml-cpu.cpp:439-447`:
   `case GGML_OP_MUL_MAT_Q4NX: case GGML_OP_MUL_MAT_ID_Q4NX:` → *"Q4NX ops run only on the HRX2 backend"*.
3. **HRX0 rejects Q4NX** — the device switch `ggml_backend_hrx_device_supports_op`
   (`ggml/src/ggml-hrx/ggml-hrx.cpp:12734`) has cases for `GGML_OP_MUL_MAT` (`:12761`) and
   `GGML_OP_MUL_MAT_ID` (`:12787`) but **no** case for `GGML_OP_MUL_MAT_Q4NX` / `GGML_OP_MUL_MAT_ID_Q4NX`,
   so it returns `false` (`:12790 default:`).
4. Therefore, with only CPU + HRX0 present, **no** buffer type can host the 280 Q4NX tensors → hard load failure.

HRX2 accepts them — `ggml-hrx2.cpp:12159` / `:12161` route `GGML_OP_MUL_MAT_Q4NX` /
`GGML_OP_MUL_MAT_ID_Q4NX` to `ggml_backend_hrx2_supports_mul_mat[_id]_q4nx_route(...)`.

> So "Q4NX does not load on HRX0" = *no HRX0 buffer type claims the Q4NX ops*, compounded by the
> unregistered type 43. Both must be fixed; the kernels below are what makes the claim executable.

---

## 3. HRX2 Q4NX kernel inventory (the set to port)

All in `ggml/src/ggml-hrx2/kernels/`, registered in `catalog/sources.json` + `catalog/artifacts.json`:

| Loom source | Lines | Exported symbol(s) | Role |
|---|---:|---|---|
| `q4nx_dequant_f32.loom` | 188 | `hrx2_q4nx_dequant_f32` | tile → F32 dequant (oracle / non-fused path) |
| `mul_mat_q4nx_fused_f32.loom` | 1157 | `hrx2_mul_mat_q4nx_fused_f32`, `..._r16`, `..._r16x8`, `..._r16x8t` | fused dequant+matmul, dense 2-D |
| `mul_mat_q4nx_fused_f32_r16w.loom` | 796 | `hrx2_mul_mat_q4nx_fused_f32_r16w` | r16, wave variant |
| `mul_mat_q4nx_fused_f32_r16wb2.loom` | 974 | `..._r16wb2` | r16, 2-col block |
| `mul_mat_q4nx_fused_f32_r16wb4c.loom` | 1289 | `..._r16wb4c` | r16, 4-col block |
| `mul_mat_q4nx_fused_f32_r16wb8.loom` | 2230 | `..._r16wb8` | r16, 8-col block |
| `mul_mat_q4nx_fused_tbl_tiled.loom` | ~ | `hrx2_mul_mat_q4nx_fused_tbl_tiled` | table-driven tiled variant |

Shared tuning knobs (all kernels): `hrx2.tuning.q4nx.workgroup_size` (`%32`, 32..1024),
`hrx2.tuning.q4nx.n_tile_cols` (1..256). Shape knobs: `k`, `rows`, `cols`, `ntokens`,
`src1_cols_count`, `nselected`.

The **MoE route** has no dedicated Loom export of its own: `MUL_MAT_ID_Q4NX` reuses the fused dense
kernel per expert slice (see `ggml-hrx2.cpp:8649` "Q4NX slice helper ... Used by both the 2-D
MUL_MAT_Q4NX (whole tensor) and MUL_MAT_ID_Q4NX (one expert per call)"), driven by
`ggml_backend_hrx2_dispatch_mul_mat_id_q4nx` (`:9420`+).

---

## 4. Q4NX format (must be reproduced bit-exactly)

From `q4nx_dequant_f32.loom` and `ggml/include/ggml.h`:

- **5120-byte tile** = one `[32 x 256]` block; weight stored **tile-major** as `[8192, n_tiles]`
  (2-D) or `[8192, tpe, n_expert]` (3-D MoE).
- Sections per tile: `[0..511]` 256 BF16 **scales** (row-major `scales[row*8 + col/32]`),
  `[512..1023]` 256 BF16 **zero-points** (same layout), `[1024..5119]` 4096 B **packed int4**.
- Two's-complement **signed int4** (`q = nib<8 ? nib : nib-16`), nibble order
  `lane = row/16, lane_row = row%16, byte_idx = lane_row/2, nib = row%2`,
  byte at `packed[lane*2048 + col*8 + byte_idx]`, val `= q*scale + zp`.
- **Clamp**: non-finite or `|scale|>100` / `|zp|>100` are zeroed. Trailer bytes (zp indices 253..255)
  are *not* zero-points and must not be used.
- Tile index for output element `(row,col)`: `tr=row/32, tc=col/256, t=tr*n_tile_cols+tc`.

The three sections are views of **one** raw blob (offsets 1024 / 0 / 512), so a slice uploads with a
single stream copy.

---

## 5. HRX0 kernel ABI conventions (target side)

HRX0 kernels are hand-written **HIP C++** (`.hip.cpp`), not Loom (`grep -rn loom ggml/src/ggml-hrx` → empty):

- Each source file defines `struct hrx_<name>_constants { long long ...; };` with a `static_assert`
  on `sizeof`, plus the `__global__` kernel. See `kernels/mul_mat_id_q4_k.hip.cpp`.
- Registration lives in `kernels/generate_hrx_kernels.py` `KERNELS[]`; each entry has
  `name`, `source`, `format` (None for HIP), `binding_count`, `parameter_count` (optional),
  `constants_size`, `workgroup_size`. The script emits `hrx_kernel_catalog.h`-shaped entries
  (`name, gfx_target, data, data_size, format, binding_count, parameter_count, constants_size, workgroup_size[3]`).
- Runtime side (`ggml-hrx.cpp`): per-op provider objects (`struct ggml_backend_hrx_op_provider`, `:57`),
  loaded by `ggml_backend_hrx_load_mul_mat_vec_providers` (`:2721`) /
  `ggml_backend_hrx_load_mul_mat_id_providers` (`:2990`); each op has a **`supports_*`** predicate and a
  **`dispatch_*`** executor, and both a **device `supports_op`** case (`:12734`) and a **compute dispatch**
  case (the switch containing `GGML_OP_MUL_MAT_ID` at `:12621`).

Existing reference analogues in HRX0:

| Purpose | HRX0 kernel | HRX0 constants struct |
|---|---|---|
| dense mat-vec | `hrx_mul_mat_vec_q4_k_f32` (`KERNELS` `:1177`, 3 bindings / 6 params / 24 B / wg256) | `ggml_backend_hrx_mul_mat_vec_constants` (`:451`, 24 B) |
| MoE mat-id | `hrx_mul_mat_id_q4_k_f32` (`KERNELS` `:969`, 4 bindings / 104 B / wg256) | `ggml_backend_hrx_mul_mat_id_q4_k_constants` (`:502`, 104 B) |

---

## 6. Kernel mapping — HRX2 → HRX0 target

Proposed new HRX0 sources/entries (names follow the HRX0 `hrx_*` convention; exact signatures are
fixed in task-2/3/4, this is the working contract):

| # | New HRX0 file | Catalog `name` | Bindings | Constants (proposed) |
|---|---|---|---|---|
| 1 | `kernels/q4nx_dequant_f32.hip.cpp` | `hrx_q4nx_dequant_f32` | `packed`(view@1024), `scales`(view@0), `zeros`(view@512), `dst` | `ncols, nrows, n_tile_cols` |
| 2 | `kernels/mul_mat_q4nx_fused_f32.hip.cpp` | `hrx_mul_mat_q4nx_fused_f32` | `src0` (Q4NX tile-major), `src1` (F32 acts), `dst` (F32) | `k, rows, cols, ntokens, n_tile_cols, src0_nb*, src1_nb*, dst_nb*` |
| 3 | `kernels/mul_mat_id_q4nx_fused_f32.hip.cpp` | `hrx_mul_mat_id_q4nx_fused_f32` | `src0` (3-D), `src1`, `ids`, `dst` | `k, tpe, n_expert, n_ids, n_tokens, n_tile_cols, *_nb*` |

Kernel 2 covers `GGML_OP_MUL_MAT_Q4NX`; kernel 3 covers `GGML_OP_MUL_MAT_ID_Q4NX` (MoE, Zaya's
`ffn_*_exps`). Kernel 1 is the dequant oracle used for per-op parity tests and any non-fused fallback.
Only **one** fused kernel variant needs to be ported for correctness (the HRX2 r16/r16wb2/b4c/b8/tbl
variants are performance tunings of the same math); pick the simplest that passes parity.

---

## 7. Exact HRX0 change sites (`ggml/src/ggml-hrx/ggml-hrx.cpp`)

| Site | Line | Change |
|---|---|---|
| device `supports_op` switch | `:12734` | add `case GGML_OP_MUL_MAT_Q4NX:` / `case GGML_OP_MUL_MAT_ID_Q4NX:` → new `ggml_backend_hrx_supports_mul_mat[_id]_q4nx(...)` |
| compute dispatch switch | near `:12621` (`GGML_OP_MUL_MAT_ID`) | add Q4NX cases → new `ggml_backend_hrx_dispatch_mul_mat[_id]_q4nx(...)` |
| provider loading | `:2721` / `:2990` | load the new catalog kernels into new provider fields |
| constants structs | near `:451` / `:502` | add `ggml_backend_hrx_mul_mat[_id]_q4nx_constants` + `static_assert` |
| catalog | `kernels/generate_hrx_kernels.py` `KERNELS[]` | add entries 1–3 (§6) |

Support predicates must mirror HRX2's element checks (`ggml-hrx2.cpp:8525-8568` for dense,
`:9420-9470` for ID): `src0->type == GGML_TYPE_Q4NX`, `src0->ne[0] == 8192`,
`src1->type == GGML_TYPE_F32`, `op->type == GGML_TYPE_F32`, `k % 256 == 0`, contiguity, and the
tile/row-count arithmetic.

---

## 8. Type-registration gap (type 43)

Separate from kernels: `b08d51080` registered `Q4NX_C43` (43) only in gguf-py. To load
`zaya1-8b-ft-q4nx.gguf`, `ggml.h`/`ggml.c` need `GGML_TYPE_Q4NX_C43 = 43` (with the same
`type_size`/`blck_size`/`type_name`/`is_quantized` traits as 42) and `GGML_TYPE_COUNT = 44`.
Decide with the user whether the port targets type 42 only (use the `-c42` models) or both.
The kernel port itself is type-agnostic — both 42 and 43 use the same 5120-byte tile layout.

---

## 9. Risks / unknowns

- **HRX0 upload path for Q4NX tensors**: HRX2 uses a view-offset binding trick (three views at
  offsets 0/512/1024 over one blob) and a host-visible scratch; HRX0's buffer/view binding
  (`hrx_buffer_binding`, `hrx_owned_buffer`) must express the same offsets or the kernel must read
  the tile directly (recommended: read the tile directly, avoiding the 3-view assembly).
- **MoE ids scratch**: HRX2 keeps a persistent host-visible `q4nx_ids` buffer
  (`ggml-hrx2.cpp:114`); HRX0's `mul_mat_id` already handles ids — reuse that path.
- **Buffer-type selection**: even after `supports_op` returns true, the loader must be able to place
  7 GB of Q4NX weights in an HRX0 buffer type; verify `ggml_backend_hrx_buffer_type_alloc_buffer`
  and the `select_weight_buft` "extra" (device-local) path.
- **Parity baseline**: HRX2 must be buildable in the same tree to produce reference logits
  (`build-both` exists).
- Not verified: whether the prebuilt `build-hrx0-latest` matches current source (it predates
  `b08d51080`); a rebuild is required before kernel tests are meaningful.

---

## 10. Mapping to goal tasks

- **task-2** → kernel #1 (`hrx_q4nx_dequant_f32`) + catalog entry; per-op dequant parity.
- **task-3** → kernel #2 + `supports/dispatch` for `GGML_OP_MUL_MAT_Q4NX`; dense parity.
- **task-4** → kernel #3 + `supports/dispatch` for `GGML_OP_MUL_MAT_ID_Q4NX`; MoE parity.
- **task-5** → rebuild HRX0+HRX2, load `zaya1-8b-ft-q4nx-*.gguf` on HRX0, decode, compare top-1 +
  logits vs HRX2/CPU, run the HRX0 test suite.
- **Open decision** (§8): scope of type-43 registration.
