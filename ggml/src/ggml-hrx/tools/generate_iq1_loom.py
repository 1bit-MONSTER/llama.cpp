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
#
# IQ1_S / IQ1_M grid lookups and decoders for motifs/dequant.loom (mode "dequant") and lane functions
# and grid fill for ops/kquant_decode_f32.loom (mode "kquant"), generated from ggml-common.h
# (iq1s_grid: MIT, the ggml authors). Every grid byte is -1, 0 or 1, so an entry is stored as a
# 16-bit code: 2 bits per value, value + 1.
# Usage: generate_iq1_loom.py ggml/src/ggml-common.h dequant|kquant > out.loom
# The output sits verbatim in kernel-corpus/kernels/loom-libs/motifs/dequant.loom (dequant) or
# kernel-corpus/kernels/loom-libs/ops/kquant_decode_f32.loom (kquant).
import re
import sys

src = open(sys.argv[1]).read()
mode = sys.argv[2]


def grid():
    m = re.search(r'GGML_TABLE_BEGIN\(uint64_t, iq1s_grid, NGRID_IQ1S\)(.*?)GGML_TABLE_END', src, re.S)
    assert m, "iq1s_grid"
    v = [int(x, 16) for x in re.findall(r'0x[0-9a-fA-F]+', m.group(1))]
    assert len(v) == 2048, len(v)
    lv = {0xff: 0, 0x00: 1, 0x01: 2}
    return [sum(lv[(e >> (8 * j)) & 255] << (2 * j) for j in range(8)) for e in v]


def lookup(fname, prefix, codes):
    n = len(codes) // 32
    o = [f'// {prefix}: {len(codes)} grid entries as 16-bit codes (2 bits per value: value + 1)',
         f'func.def inline @{fname}(%grid_index: i32) -> (i32) {{',
         '  %c5_i32 = scalar.constant 5 : i32', '  %c31_i32 = scalar.constant 31 : i32']
    for k in range(1, n):
        o.append(f'  %chunk_id{k} = scalar.constant {k} : i32')
    o += ['  %chunk_i32 = scalar.shrui %grid_index, %c5_i32 : i32',
          '  %lane_i32 = scalar.andi %grid_index, %c31_i32 : i32',
          '  %codes = vector.from_elements %lane_i32 : vector<1xi32>']
    for k in range(n):
        names = []
        for i in range(32):
            nm = f'%{prefix}_{k}_{i}'
            o.append(f'  {nm} = scalar.constant {float(codes[32 * k + i])} : f32')
            names.append(nm)
        o.append(f'  %{prefix}_{k} = vector.from_elements {", ".join(names)} : vector<32xf32>')
    prev = f'%{prefix}_0'
    for k in range(1, n):
        o.append(f'  %is_chunk{k} = scalar.cmpi eq, %chunk_i32, %chunk_id{k} : i32')
        o.append(f'  %sel{k} = scf.select %is_chunk{k}, %{prefix}_{k}, {prev} : vector<32xf32>')
        prev = f'%sel{k}'
    o += [f'  %v = vector.table.lookup {prev}[%codes] : vector<32xf32>, vector<1xi32> -> vector<1xf32>',
          '  %v_f32 = vector.extract %v[0] : vector<1xf32> -> f32',
          '  %code = scalar.fptoui %v_f32 : f32 to i32',
          '  func.return %code : i32', '}', '']
    return '\n'.join(o)


dequant_common = '''
// four values 4 (p % 2) .. +3 of an IQ1 grid code: ((code >> 2j) & 3) - 1 + delta, times %scale
func.def inline @ggml_iq1_code_vector4(%code: i32, %half: i32, %delta: f32, %scale: f32) -> (vector<4xf32>) {
  %c1_i32 = scalar.constant 1 : i32
  %c3_i32 = scalar.constant 3 : i32
  %c8_i32 = scalar.constant 8 : i32
  %c2v = scalar.constant 2 : i32
  %c4v = scalar.constant 4 : i32
  %c6v = scalar.constant 6 : i32
  %c0v = scalar.constant 0 : i32
  %base = scalar.muli %half, %c8_i32 : i32
  %cv = vector.splat %code : vector<4xi32>
  %bv = vector.splat %base : vector<4xi32>
  %sh0 = vector.from_elements %c0v, %c2v, %c4v, %c6v : vector<4xi32>
  %sh = vector.addi %sh0, %bv : vector<4xi32>
  %s = vector.shrui %cv, %sh : vector<4xi32>
  %three = vector.splat %c3_i32 : vector<4xi32>
  %one = vector.splat %c1_i32 : vector<4xi32>
  %c = vector.andi %s, %three : vector<4xi32>
  %t = vector.subi %c, %one : vector<4xi32>
  %tf = vector.sitofp %t : vector<4xi32> to vector<4xf32>
  %dv = vector.splat %delta : vector<4xf32>
  %sv = vector.splat %scale : vector<4xf32>
  %td = vector.addf %tf, %dv : vector<4xf32>
  %result = vector.mulf %td, %sv : vector<4xf32>
  func.return %result : vector<4xf32>
}

// IQ1_S (50 bytes: d, qs[32], qh[8] u16; dequantize_row_iq1_s). Group g: qh[g] holds three 3-bit
// high index parts (slot l at bit 3l), a 3-bit scale at bit 12 and the delta sign at bit 15:
// value = d * (2 s + 1) * (grid + delta), delta = +-0.125. Packet p covers slot p / 2.
func.def inline @ggml_iq1s_f32_vector4(%weight: buffer, %row_byte_base: offset, %iq1_block: index, %iq1_group: index, %packet: index) -> (vector<4xf32>) {
  %c0 = index.constant 0 : index
  %c2 = index.constant 2 : index
  %c4 = index.constant 4 : index
  %c17 = index.constant 17 : index
  %c1_i32 = scalar.constant 1 : i32
  %c3_i32 = scalar.constant 3 : i32
  %c7_i32 = scalar.constant 7 : i32
  %c8_i32 = scalar.constant 8 : i32
  %c12_i32 = scalar.constant 12 : i32
  %c32768_i32 = scalar.constant 32768 : i32
  %c0_i32 = scalar.constant 0 : i32
  %pos_delta = scalar.constant 0.125 : f32
  %neg_delta = scalar.constant -0.125 : f32
  %block_bytes = index.constant 50 : offset
  %block_byte_add = index.scale %iq1_block, %block_bytes : index, offset -> offset
  %block_byte_base = index.add %row_byte_base, %block_byte_add : offset
  %hv = buffer.view %weight[%block_byte_base] : buffer -> view<25xf16>
  %wv = buffer.view %weight[%block_byte_base] : buffer -> view<25xi16>
  %bv = buffer.view %weight[%block_byte_base] : buffer -> view<50xi8>
  %g = index.assume %iq1_group [range(%iq1_group, 0, 7)] : index
  %p = index.assume %packet [range(%packet, 0, 7)] : index
  %slot = index.div %p, %c2 : index
  %half = index.rem %p, %c2 : index
  %g4 = index.mul %g, %c4 : index
  %qs_at0 = index.add %g4, %c2 : index
  %qs_at = index.add %qs_at0, %slot : index
  %qh_at = index.add %c17, %g : index
  %d_f16 = view.load %hv[%c0] : view<25xf16> -> f16
  %d = scalar.extf %d_f16 : f16 to f32
  %qs_i8 = view.load %bv[%qs_at] : view<50xi8> -> i8
  %qs = scalar.extui %qs_i8 : i8 to i32
  %qh_i16 = view.load %wv[%qh_at] : view<25xi16> -> i16
  %qh = scalar.extui %qh_i16 : i16 to i32
  %slot_i32 = index.cast %slot : index to i32
  %hsh = scalar.muli %slot_i32, %c3_i32 : i32
  %hi0 = scalar.shrui %qh, %hsh : i32
  %hi1 = scalar.andi %hi0, %c7_i32 : i32
  %hi = scalar.shli %hi1, %c8_i32 : i32
  %gi = scalar.ori %qs, %hi : i32
  %s0 = scalar.shrui %qh, %c12_i32 : i32
  %s = scalar.andi %s0, %c7_i32 : i32
  %s2 = scalar.addi %s, %s : i32
  %s21 = scalar.addi %s2, %c1_i32 : i32
  %sf = scalar.uitofp %s21 : i32 to f32
  %scale = scalar.mulf %d, %sf : f32
  %neg_bit = scalar.andi %qh, %c32768_i32 : i32
  %neg = scalar.cmpi ne, %neg_bit, %c0_i32 : i32
  %delta = scf.select %neg, %neg_delta, %pos_delta : f32
  %code = func.call @ggml_iq1s_grid_code_i32(%gi) : (i32) -> (i32)
  %half_i32 = index.cast %half : index to i32
  %result = func.call @ggml_iq1_code_vector4(%code, %half_i32, %delta, %scale) : (i32, i32, f32, f32) -> (vector<4xf32>)
  func.return %result : vector<4xf32>
}

func.def inline @ggml_iq1s_f16_vector4(%weight: buffer, %row_byte_base: offset, %iq1_block: index, %iq1_group: index, %packet: index) -> (vector<4xf16>) {
  %values_f32 = func.call @ggml_iq1s_f32_vector4(%weight, %row_byte_base, %iq1_block, %iq1_group, %packet) : (buffer, offset, index, index, index) -> (vector<4xf32>)
  %values = vector.fptrunc %values_f32 : vector<4xf32> to vector<4xf16>
  func.return %values : vector<4xf16>
}

// IQ1_M's fp16 block scale: the top nibbles of its four u16 scale words (bytes 48..55).
func.def inline @ggml_iq1m_block_scale(%weight: buffer, %block_byte_base: offset) -> (f32) {
  %wv = buffer.view %weight[%block_byte_base] : buffer -> view<28xi16>
  %c24 = index.constant 24 : index
  %c25 = index.constant 25 : index
  %c26 = index.constant 26 : index
  %c27 = index.constant 27 : index
  %c4_i32 = scalar.constant 4 : i32
  %c8_i32 = scalar.constant 8 : i32
  %c12_i32 = scalar.constant 12 : i32
  %c240_i32 = scalar.constant 240 : i32
  %c3840_i32 = scalar.constant 3840 : i32
  %c61440_i32 = scalar.constant 61440 : i32
  %c255_i32 = scalar.constant 255 : i32
  %w0_i16 = view.load %wv[%c24] : view<28xi16> -> i16
  %w1_i16 = view.load %wv[%c25] : view<28xi16> -> i16
  %w2_i16 = view.load %wv[%c26] : view<28xi16> -> i16
  %w3_i16 = view.load %wv[%c27] : view<28xi16> -> i16
  %w0 = scalar.extui %w0_i16 : i16 to i32
  %w1 = scalar.extui %w1_i16 : i16 to i32
  %w2 = scalar.extui %w2_i16 : i16 to i32
  %w3 = scalar.extui %w3_i16 : i16 to i32
  %n0 = scalar.shrui %w0, %c12_i32 : i32
  %n1a = scalar.shrui %w1, %c8_i32 : i32
  %n1 = scalar.andi %n1a, %c240_i32 : i32
  %n2a = scalar.shrui %w2, %c4_i32 : i32
  %n2 = scalar.andi %n2a, %c3840_i32 : i32
  %n3 = scalar.andi %w3, %c61440_i32 : i32
  %u01 = scalar.ori %n0, %n1 : i32
  %u012 = scalar.ori %u01, %n2 : i32
  %u = scalar.ori %u012, %n3 : i32
  %lo = scalar.andi %u, %c255_i32 : i32
  %hi = scalar.shrui %u, %c8_i32 : i32
  %lo8 = scalar.trunci %lo : i32 to i8
  %hi8 = scalar.trunci %hi : i32 to i8
  %bytes = vector.from_elements %lo8, %hi8 : vector<2xi8>
  %h = vector.bitcast %bytes : vector<2xi8> to vector<1xf16>
  %h0 = vector.extract %h[0] : vector<1xf16> -> f16
  %f = scalar.extf %h0 : f16 to f32
  func.return %f : f32
}

// IQ1_M (56 bytes: qs[32], qh[16], scales[8]; dequantize_row_iq1_m). Slot l of group g: qh byte
// 2g + l / 2 holds the high index bits (bits 0..2 for even l, 4..6 for odd) and the delta sign
// (bit 3 / bit 7); the 3-bit scale of slots 0-1 / 2-3 sits at bit 6 (g % 2) / +3 of u16 scale
// word g / 2: value = d * (2 s + 1) * (grid + delta).
func.def inline @ggml_iq1m_f32_vector4(%weight: buffer, %row_byte_base: offset, %iq1_block: index, %iq1_group: index, %packet: index) -> (vector<4xf32>) {
  %c2 = index.constant 2 : index
  %c4 = index.constant 4 : index
  %c24 = index.constant 24 : index
  %c32 = index.constant 32 : index
  %c0_i32 = scalar.constant 0 : i32
  %c1_i32 = scalar.constant 1 : i32
  %c3_i32 = scalar.constant 3 : i32
  %c4_i32 = scalar.constant 4 : i32
  %c6_i32 = scalar.constant 6 : i32
  %c7_i32 = scalar.constant 7 : i32
  %c8_i32 = scalar.constant 8 : i32
  %c1792_i32 = scalar.constant 1792 : i32
  %pos_delta = scalar.constant 0.125 : f32
  %neg_delta = scalar.constant -0.125 : f32
  %block_bytes = index.constant 56 : offset
  %block_byte_add = index.scale %iq1_block, %block_bytes : index, offset -> offset
  %block_byte_base = index.add %row_byte_base, %block_byte_add : offset
  %wv = buffer.view %weight[%block_byte_base] : buffer -> view<28xi16>
  %bv = buffer.view %weight[%block_byte_base] : buffer -> view<56xi8>
  %g = index.assume %iq1_group [range(%iq1_group, 0, 7)] : index
  %p = index.assume %packet [range(%packet, 0, 7)] : index
  %slot = index.div %p, %c2 : index
  %half = index.rem %p, %c2 : index
  %pair = index.div %slot, %c2 : index
  %odd = index.rem %slot, %c2 : index
  %g4 = index.mul %g, %c4 : index
  %qs_at = index.add %g4, %slot : index
  %g2 = index.add %g, %g : index
  %qh_at0 = index.add %c32, %g2 : index
  %qh_at = index.add %qh_at0, %pair : index
  %gh = index.div %g, %c2 : index
  %gp = index.rem %g, %c2 : index
  %sc_at = index.add %c24, %gh : index
  %d = func.call @ggml_iq1m_block_scale(%weight, %block_byte_base) : (buffer, offset) -> (f32)
  %qs_i8 = view.load %bv[%qs_at] : view<56xi8> -> i8
  %qs = scalar.extui %qs_i8 : i8 to i32
  %qh_i8 = view.load %bv[%qh_at] : view<56xi8> -> i8
  %qh = scalar.extui %qh_i8 : i8 to i32
  %odd_i32 = index.cast %odd : index to i32
  %odd4 = scalar.muli %odd_i32, %c4_i32 : i32
  %qh_n = scalar.shrui %qh, %odd4 : i32
  %hi0 = scalar.shli %qh_n, %c8_i32 : i32
  %hi = scalar.andi %hi0, %c1792_i32 : i32
  %gi = scalar.ori %qs, %hi : i32
  %neg_bit0 = scalar.shrui %qh_n, %c3_i32 : i32
  %neg_bit = scalar.andi %neg_bit0, %c1_i32 : i32
  %neg = scalar.cmpi ne, %neg_bit, %c0_i32 : i32
  %delta = scf.select %neg, %neg_delta, %pos_delta : f32
  %sc_i16 = view.load %wv[%sc_at] : view<28xi16> -> i16
  %sc = scalar.extui %sc_i16 : i16 to i32
  %gp_i32 = index.cast %gp : index to i32
  %pair_i32 = index.cast %pair : index to i32
  %ssh0 = scalar.muli %gp_i32, %c6_i32 : i32
  %ssh1 = scalar.muli %pair_i32, %c3_i32 : i32
  %ssh = scalar.addi %ssh0, %ssh1 : i32
  %s0 = scalar.shrui %sc, %ssh : i32
  %s = scalar.andi %s0, %c7_i32 : i32
  %s2 = scalar.addi %s, %s : i32
  %s21 = scalar.addi %s2, %c1_i32 : i32
  %sf = scalar.uitofp %s21 : i32 to f32
  %scale = scalar.mulf %d, %sf : f32
  %code = func.call @ggml_iq1s_grid_code_i32(%gi) : (i32) -> (i32)
  %half_i32 = index.cast %half : index to i32
  %result = func.call @ggml_iq1_code_vector4(%code, %half_i32, %delta, %scale) : (i32, i32, f32, f32) -> (vector<4xf32>)
  func.return %result : vector<4xf32>
}

func.def inline @ggml_iq1m_f16_vector4(%weight: buffer, %row_byte_base: offset, %iq1_block: index, %iq1_group: index, %packet: index) -> (vector<4xf16>) {
  %values_f32 = func.call @ggml_iq1m_f32_vector4(%weight, %row_byte_base, %iq1_block, %iq1_group, %packet) : (buffer, offset, index, index, index) -> (vector<4xf32>)
  %values = vector.fptrunc %values_f32 : vector<4xf32> to vector<4xf16>
  func.return %values : vector<4xf16>
}
'''


def fill(fname, codes):
    words = [codes[2 * w] | (codes[2 * w + 1] << 16) for w in range(len(codes) // 2)]
    words = [w - (1 << 32) if w >= 1 << 31 else w for w in words]  # i32 constants are signed
    per = len(words) // 4
    o = [f'// Stages the IQ1 grid as {len(words)} words (two 16-bit codes each); subgroup %chunk (0..3) writes',
         f'// words {per} %chunk .. +{per - 1}.',
         f'func.def inline @{fname}(%grid: buffer, %chunk: index) {{',
         '  %zero_offset = index.constant 0 : offset',
         f'  %gv = buffer.view %grid[%zero_offset] : buffer -> view<{len(words)}xi32>']
    for c in range(4):
        o += [f'  %k{c} = index.constant {c} : index', f'  %is{c} = index.cmp eq, %chunk, %k{c} : index', f'  scf.if %is{c} {{']
        for q in range(per // 4):
            e = per * c + 4 * q
            for i in range(4):
                o.append(f'    %v{c}_{q}_{i} = scalar.constant {words[e + i]} : i32')
            o.append(f'    %w{c}_{q} = vector.from_elements %v{c}_{q}_0, %v{c}_{q}_1, %v{c}_{q}_2, %v{c}_{q}_3 : vector<4xi32>')
            o.append(f'    %o{c}_{q} = index.constant {e} : index')
            o.append(f'    vector.store %w{c}_{q}, %gv[%o{c}_{q}] : vector<4xi32>, view<{len(words)}xi32>')
        o.append('  }')
    o += ['  func.return', '}', '']
    return '\n'.join(o)


kquant_common = '''
// IQ1_M's fp16 block scale (as motifs/dequant.loom's ggml_iq1m_block_scale).
func.def inline @ggml_kquant_iq1m_block_scale(%weight: buffer, %block_byte_base: offset) -> (f32) {
  %wv = buffer.view %weight[%block_byte_base] : buffer -> view<28xi16>
  %c24 = index.constant 24 : index
  %c25 = index.constant 25 : index
  %c26 = index.constant 26 : index
  %c27 = index.constant 27 : index
  %c4_i32 = scalar.constant 4 : i32
  %c8_i32 = scalar.constant 8 : i32
  %c12_i32 = scalar.constant 12 : i32
  %c240_i32 = scalar.constant 240 : i32
  %c3840_i32 = scalar.constant 3840 : i32
  %c61440_i32 = scalar.constant 61440 : i32
  %c255_i32 = scalar.constant 255 : i32
  %w0_i16 = view.load %wv[%c24] : view<28xi16> -> i16
  %w1_i16 = view.load %wv[%c25] : view<28xi16> -> i16
  %w2_i16 = view.load %wv[%c26] : view<28xi16> -> i16
  %w3_i16 = view.load %wv[%c27] : view<28xi16> -> i16
  %w0 = scalar.extui %w0_i16 : i16 to i32
  %w1 = scalar.extui %w1_i16 : i16 to i32
  %w2 = scalar.extui %w2_i16 : i16 to i32
  %w3 = scalar.extui %w3_i16 : i16 to i32
  %n0 = scalar.shrui %w0, %c12_i32 : i32
  %n1a = scalar.shrui %w1, %c8_i32 : i32
  %n1 = scalar.andi %n1a, %c240_i32 : i32
  %n2a = scalar.shrui %w2, %c4_i32 : i32
  %n2 = scalar.andi %n2a, %c3840_i32 : i32
  %n3 = scalar.andi %w3, %c61440_i32 : i32
  %u01 = scalar.ori %n0, %n1 : i32
  %u012 = scalar.ori %u01, %n2 : i32
  %u = scalar.ori %u012, %n3 : i32
  %lo = scalar.andi %u, %c255_i32 : i32
  %hi = scalar.shrui %u, %c8_i32 : i32
  %lo8 = scalar.trunci %lo : i32 to i8
  %hi8 = scalar.trunci %hi : i32 to i8
  %bytes = vector.from_elements %lo8, %hi8 : vector<2xi8>
  %h = vector.bitcast %bytes : vector<2xi8> to vector<1xf16>
  %h0 = vector.extract %h[0] : vector<1xf16> -> f16
  %f = scalar.extf %h0 : f16 to f32
  func.return %f : f32
}

// IQ1_S / IQ1_M: lane l16 owns group l16 / 2, slots 2 (l16 % 2) and +1: values 32 (l16 / 2) +
// 16 (l16 % 2) .. +15, one scale. The grid (2048 16-bit codes, 2 bits per value: value + 1) is
// staged in workgroup memory two codes per word.
func.def inline @ggml_kquant_iq1_code(%grid: buffer, %index: i32) -> (i32) {
  %zero_offset = index.constant 0 : offset
  %gv = buffer.view %grid[%zero_offset] : buffer -> view<1024xi32>
  %c1_i32 = scalar.constant 1 : i32
  %c4_i32 = scalar.constant 4 : i32
  %c65535_i32 = scalar.constant 65535 : i32
  %w_i32 = scalar.shrui %index, %c1_i32 : i32
  %w_x = index.cast %w_i32 : i32 to index
  %w_b = index.assume %w_x [range(%w_x, 0, 1023)] : index
  %word = view.load %gv[%w_b] : view<1024xi32> -> i32
  %odd = scalar.andi %index, %c1_i32 : i32
  %sh = scalar.shli %odd, %c4_i32 : i32
  %shifted = scalar.shrui %word, %sh : i32
  %code = scalar.andi %shifted, %c65535_i32 : i32
  func.return %code : i32
}

func.def inline @ggml_kquant_iq1_slot_values(%code: i32, %delta: f32) -> (vector<8xf32>) {
  %c1_i32 = scalar.constant 1 : i32
  %s0 = scalar.constant 0 : i32
  %s2 = scalar.constant 2 : i32
  %s3 = scalar.constant 3 : i32
  %s4 = scalar.constant 4 : i32
  %s6 = scalar.constant 6 : i32
  %s8 = scalar.constant 8 : i32
  %s10 = scalar.constant 10 : i32
  %s12 = scalar.constant 12 : i32
  %s14 = scalar.constant 14 : i32
  %shift2 = vector.from_elements %s0, %s2, %s4, %s6, %s8, %s10, %s12, %s14 : vector<8xi32>
  %three = vector.splat %s3 : vector<8xi32>
  %one = vector.splat %c1_i32 : vector<8xi32>
  %cv = vector.splat %code : vector<8xi32>
  %cs = vector.shrui %cv, %shift2 : vector<8xi32>
  %c = vector.andi %cs, %three : vector<8xi32>
  %t = vector.subi %c, %one : vector<8xi32>
  %tf = vector.sitofp %t : vector<8xi32> to vector<8xf32>
  %dv = vector.splat %delta : vector<8xf32>
  %v = vector.addf %tf, %dv : vector<8xf32>
  func.return %v : vector<8xf32>
}

// IQ1_S (50 bytes: d, qs[32], qh[8] u16): qh[g] = high index bits (slot l at bit 3l), scale s at
// bit 12, delta sign at bit 15: value = d * (2 s + 1) * (grid + delta).
func.def inline @ggml_kquant_iq1s_lane_parts(%grid: buffer, %weight: buffer, %row_base: offset, %block: index, %lane16: index) -> (vector<16xf32>, f32, index) {
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c2 = index.constant 2 : index
  %c4 = index.constant 4 : index
  %c16 = index.constant 16 : index
  %c17 = index.constant 17 : index
  %c32 = index.constant 32 : index
  %c0_i32 = scalar.constant 0 : i32
  %c1_i32 = scalar.constant 1 : i32
  %c3_i32 = scalar.constant 3 : i32
  %c7_i32 = scalar.constant 7 : i32
  %c8_i32 = scalar.constant 8 : i32
  %c12_i32 = scalar.constant 12 : i32
  %c32768_i32 = scalar.constant 32768 : i32
  %pos_delta = scalar.constant 0.125 : f32
  %neg_delta = scalar.constant -0.125 : f32
  %block_bytes = index.constant 50 : offset
  %l = index.assume %lane16 [range(%lane16, 0, 15)] : index
  %g = index.div %l, %c2 : index
  %h = index.rem %l, %c2 : index
  %sa = index.mul %h, %c2 : index
  %sb = index.add %sa, %c1 : index
  %block_add = index.scale %block, %block_bytes : index, offset -> offset
  %block_base = index.add %row_base, %block_add : offset
  %hv = buffer.view %weight[%block_base] : buffer -> view<25xf16>
  %wv = buffer.view %weight[%block_base] : buffer -> view<25xi16>
  %bv = buffer.view %weight[%block_base] : buffer -> view<50xi8>
  %d_f16 = view.load %hv[%c0] : view<25xf16> -> f16
  %d = scalar.extf %d_f16 : f16 to f32
  %qh_at = index.add %c17, %g : index
  %qh_i16 = view.load %wv[%qh_at] : view<25xi16> -> i16
  %qh = scalar.extui %qh_i16 : i16 to i32
  %g4 = index.mul %g, %c4 : index
  %qs0 = index.add %g4, %c2 : index
  %qa_at = index.add %qs0, %sa : index
  %qb_at = index.add %qs0, %sb : index
  %qa_i8 = view.load %bv[%qa_at] : view<50xi8> -> i8
  %qb_i8 = view.load %bv[%qb_at] : view<50xi8> -> i8
  %qa = scalar.extui %qa_i8 : i8 to i32
  %qb = scalar.extui %qb_i8 : i8 to i32
  %sa_i32 = index.cast %sa : index to i32
  %sb_i32 = index.cast %sb : index to i32
  %sha = scalar.muli %sa_i32, %c3_i32 : i32
  %shb = scalar.muli %sb_i32, %c3_i32 : i32
  %ha0 = scalar.shrui %qh, %sha : i32
  %hb0 = scalar.shrui %qh, %shb : i32
  %ha1 = scalar.andi %ha0, %c7_i32 : i32
  %hb1 = scalar.andi %hb0, %c7_i32 : i32
  %ha = scalar.shli %ha1, %c8_i32 : i32
  %hb = scalar.shli %hb1, %c8_i32 : i32
  %ia = scalar.ori %qa, %ha : i32
  %ib = scalar.ori %qb, %hb : i32
  %code_a = func.call @ggml_kquant_iq1_code(%grid, %ia) : (buffer, i32) -> (i32)
  %code_b = func.call @ggml_kquant_iq1_code(%grid, %ib) : (buffer, i32) -> (i32)
  %s0 = scalar.shrui %qh, %c12_i32 : i32
  %s = scalar.andi %s0, %c7_i32 : i32
  %s2 = scalar.addi %s, %s : i32
  %s21 = scalar.addi %s2, %c1_i32 : i32
  %sf = scalar.uitofp %s21 : i32 to f32
  %scale = scalar.mulf %d, %sf : f32
  %neg_bit = scalar.andi %qh, %c32768_i32 : i32
  %neg = scalar.cmpi ne, %neg_bit, %c0_i32 : i32
  %delta = scf.select %neg, %neg_delta, %pos_delta : f32
  %va = func.call @ggml_kquant_iq1_slot_values(%code_a, %delta) : (i32, f32) -> (vector<8xf32>)
  %vb = func.call @ggml_kquant_iq1_slot_values(%code_b, %delta) : (i32, f32) -> (vector<8xf32>)
  %v = func.call @ggml_kquant_iq2_join16(%va, %vb) : (vector<8xf32>, vector<8xf32>) -> (vector<16xf32>)
  %g32 = index.mul %g, %c32 : index
  %h16 = index.mul %h, %c16 : index
  %p0 = index.add %g32, %h16 : index
  func.return %v, %scale, %p0 : vector<16xf32>, f32, index
}

// IQ1_M (56 bytes: qs[32], qh[16], scales[8]): the lane's two slots share qh byte 2g + h (index
// high bits 0..2 / 4..6, delta signs bit 3 / 7) and the 3-bit scale at bit 6 (g % 2) + 3h of u16
// scale word g / 2; d is the fp16 made of the four scale words' top nibbles.
func.def inline @ggml_kquant_iq1m_lane_parts(%grid: buffer, %weight: buffer, %row_base: offset, %block: index, %lane16: index) -> (vector<16xf32>, f32, index) {
  %c1 = index.constant 1 : index
  %c2 = index.constant 2 : index
  %c4 = index.constant 4 : index
  %c16 = index.constant 16 : index
  %c24 = index.constant 24 : index
  %c32 = index.constant 32 : index
  %c0_i32 = scalar.constant 0 : i32
  %c1_i32 = scalar.constant 1 : i32
  %c3_i32 = scalar.constant 3 : i32
  %c4_i32 = scalar.constant 4 : i32
  %c6_i32 = scalar.constant 6 : i32
  %c7_i32 = scalar.constant 7 : i32
  %c8_i32 = scalar.constant 8 : i32
  %c128_i32 = scalar.constant 128 : i32
  %c1792_i32 = scalar.constant 1792 : i32
  %pos_delta = scalar.constant 0.125 : f32
  %neg_delta = scalar.constant -0.125 : f32
  %block_bytes = index.constant 56 : offset
  %l = index.assume %lane16 [range(%lane16, 0, 15)] : index
  %g = index.div %l, %c2 : index
  %h = index.rem %l, %c2 : index
  %sa = index.mul %h, %c2 : index
  %sb = index.add %sa, %c1 : index
  %block_add = index.scale %block, %block_bytes : index, offset -> offset
  %block_base = index.add %row_base, %block_add : offset
  %wv = buffer.view %weight[%block_base] : buffer -> view<28xi16>
  %bv = buffer.view %weight[%block_base] : buffer -> view<56xi8>
  %d = func.call @ggml_kquant_iq1m_block_scale(%weight, %block_base) : (buffer, offset) -> (f32)
  %g4 = index.mul %g, %c4 : index
  %qa_at = index.add %g4, %sa : index
  %qb_at = index.add %g4, %sb : index
  %qa_i8 = view.load %bv[%qa_at] : view<56xi8> -> i8
  %qb_i8 = view.load %bv[%qb_at] : view<56xi8> -> i8
  %qa = scalar.extui %qa_i8 : i8 to i32
  %qb = scalar.extui %qb_i8 : i8 to i32
  %g2 = index.add %g, %g : index
  %qh_at0 = index.add %c32, %g2 : index
  %qh_at = index.add %qh_at0, %h : index
  %qh_i8 = view.load %bv[%qh_at] : view<56xi8> -> i8
  %qh = scalar.extui %qh_i8 : i8 to i32
  %ha0 = scalar.shli %qh, %c8_i32 : i32
  %ha = scalar.andi %ha0, %c1792_i32 : i32
  %hb0 = scalar.shli %qh, %c4_i32 : i32
  %hb = scalar.andi %hb0, %c1792_i32 : i32
  %ia = scalar.ori %qa, %ha : i32
  %ib = scalar.ori %qb, %hb : i32
  %code_a = func.call @ggml_kquant_iq1_code(%grid, %ia) : (buffer, i32) -> (i32)
  %code_b = func.call @ggml_kquant_iq1_code(%grid, %ib) : (buffer, i32) -> (i32)
  %na0 = scalar.andi %qh, %c8_i32 : i32
  %nb0 = scalar.andi %qh, %c128_i32 : i32
  %na = scalar.cmpi ne, %na0, %c0_i32 : i32
  %nb = scalar.cmpi ne, %nb0, %c0_i32 : i32
  %delta_a = scf.select %na, %neg_delta, %pos_delta : f32
  %delta_b = scf.select %nb, %neg_delta, %pos_delta : f32
  %gh = index.div %g, %c2 : index
  %gp = index.rem %g, %c2 : index
  %sc_at = index.add %c24, %gh : index
  %sc_i16 = view.load %wv[%sc_at] : view<28xi16> -> i16
  %sc = scalar.extui %sc_i16 : i16 to i32
  %gp_i32 = index.cast %gp : index to i32
  %h_i32 = index.cast %h : index to i32
  %ssh0 = scalar.muli %gp_i32, %c6_i32 : i32
  %ssh1 = scalar.muli %h_i32, %c3_i32 : i32
  %ssh = scalar.addi %ssh0, %ssh1 : i32
  %s0 = scalar.shrui %sc, %ssh : i32
  %s = scalar.andi %s0, %c7_i32 : i32
  %s2 = scalar.addi %s, %s : i32
  %s21 = scalar.addi %s2, %c1_i32 : i32
  %sf = scalar.uitofp %s21 : i32 to f32
  %scale = scalar.mulf %d, %sf : f32
  %va = func.call @ggml_kquant_iq1_slot_values(%code_a, %delta_a) : (i32, f32) -> (vector<8xf32>)
  %vb = func.call @ggml_kquant_iq1_slot_values(%code_b, %delta_b) : (i32, f32) -> (vector<8xf32>)
  %v = func.call @ggml_kquant_iq2_join16(%va, %vb) : (vector<8xf32>, vector<8xf32>) -> (vector<16xf32>)
  %g32 = index.mul %g, %c32 : index
  %h16 = index.mul %h, %c16 : index
  %p0 = index.add %g32, %h16 : index
  func.return %v, %scale, %p0 : vector<16xf32>, f32, index
}
'''


def dot_and_weights(fmt):
    return f'''
func.def inline @ggml_kquant_{fmt}_lane_dot(%grid: buffer, %weight: buffer, %input: buffer, %row_base: offset, %block: index, %lane16: index) -> (f32) {{
  %zero_scalar = scalar.constant 0.0 : f32
  %x_block_bytes = index.constant 1024 : offset
  %v, %scale, %p = func.call @ggml_kquant_{fmt}_lane_parts(%grid, %weight, %row_base, %block, %lane16) : (buffer, buffer, offset, index, index) -> (vector<16xf32>, f32, index)
  %x_base = index.scale %block, %x_block_bytes : index, offset -> offset
  %xv = buffer.view %input[%x_base] : buffer -> view<256xf32>
  %x = vector.load %xv[%p] : view<256xf32> -> vector<16xf32>
  %vx = vector.mulf %v, %x : vector<16xf32>
  %sum = vector.reduce<addf> %vx, %zero_scalar : vector<16xf32>, f32
  %result = scalar.mulf %scale, %sum : f32
  func.return %result : f32
}}

func.def inline @ggml_kquant_{fmt}_lane_weights(%grid: buffer, %weight: buffer, %row_base: offset, %block: index, %lane16: index) -> (vector<16xf32>, index, index, index, index) {{
  %c4 = index.constant 4 : index
  %c8 = index.constant 8 : index
  %c12 = index.constant 12 : index
  %v, %scale, %p0 = func.call @ggml_kquant_{fmt}_lane_parts(%grid, %weight, %row_base, %block, %lane16) : (buffer, buffer, offset, index, index) -> (vector<16xf32>, f32, index)
  %sv = vector.splat %scale : vector<16xf32>
  %w = vector.mulf %v, %sv : vector<16xf32>
  %p1 = index.add %p0, %c4 : index
  %p2 = index.add %p0, %c8 : index
  %p3 = index.add %p0, %c12 : index
  func.return %w, %p0, %p1, %p2, %p3 : vector<16xf32>, index, index, index, index
}}
'''


if mode == "dequant":
    out = lookup('ggml_iq1s_grid_code_i32', 'iq1s_code', grid()) + dequant_common
elif mode == "kquant":
    out = fill('ggml_kquant_iq1s_grid_fill', grid()) + kquant_common + dot_and_weights('iq1s') + dot_and_weights('iq1m')
else:
    sys.exit(f"unknown mode {mode}")
sys.stdout.write(out)
