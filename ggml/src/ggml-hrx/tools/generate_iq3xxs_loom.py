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
# IQ3_XXS grid lookups and decoders for motifs/dequant.loom (mode "dequant") and lane functions and
# grid fill for ops/kquant_decode_f32.loom (mode "kquant"), generated from ggml-common.h (iq3xxs_grid:
# MIT, the ggml authors). Every grid byte is one of 4, 12, 20, 28, 36, 44, 52, 62, so an entry (4
# values) is stored as a 12-bit code: 3 bits per value, level L = 0..7, value 4 + 8 L (+ 2 for L = 7).
# Usage: generate_iq3xxs_loom.py ggml/src/ggml-common.h dequant|kquant > out.loom
# The output sits verbatim in kernel-corpus/kernels/loom-libs/motifs/dequant.loom (dequant) or
# kernel-corpus/kernels/loom-libs/ops/kquant_decode_f32.loom (kquant).
import re
import sys

src = open(sys.argv[1]).read()
mode = sys.argv[2]
LEVELS = {4: 0, 12: 1, 20: 2, 28: 3, 36: 4, 44: 5, 52: 6, 62: 7}


def grid():
    m = re.search(r'GGML_TABLE_BEGIN\(uint32_t, iq3xxs_grid, 256\)(.*?)GGML_TABLE_END', src, re.S)
    assert m, "iq3xxs_grid"
    v = [int(x, 16) for x in re.findall(r'0x[0-9a-fA-F]+', m.group(1))]
    assert len(v) == 256, len(v)
    return [sum(LEVELS[(e >> (8 * j)) & 255] << (3 * j) for j in range(4)) for e in v]


def lookup(fname, prefix, codes):
    n = len(codes) // 32
    o = [f'// {prefix}: {len(codes)} grid entries as 12-bit codes (3 bits per value: level L, value 4 + 8 L, 62 for L = 7)',
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
// the four values of a 12-bit IQ3_XXS grid code, with sign bits 4 half .. +3 of %signs8, times %scale
func.def inline @ggml_iq3xxs_code_vector4(%code: i32, %half: i32, %signs8: i32, %scale: f32) -> (vector<4xf32>) {
  %c1_i32 = scalar.constant 1 : i32
  %c2_i32 = scalar.constant 2 : i32
  %c4_i32 = scalar.constant 4 : i32
  %c7_i32 = scalar.constant 7 : i32
  %c8_i32 = scalar.constant 8 : i32
  %s0 = scalar.constant 0 : i32
  %s3 = scalar.constant 3 : i32
  %s6 = scalar.constant 6 : i32
  %s9 = scalar.constant 9 : i32
  %shift3 = vector.from_elements %s0, %s3, %s6, %s9 : vector<4xi32>
  %cv = vector.splat %code : vector<4xi32>
  %lvs = vector.shrui %cv, %shift3 : vector<4xi32>
  %seven = vector.splat %c7_i32 : vector<4xi32>
  %lv = vector.andi %lvs, %seven : vector<4xi32>
  %eight = vector.splat %c8_i32 : vector<4xi32>
  %four = vector.splat %c4_i32 : vector<4xi32>
  %lv8 = vector.muli %lv, %eight : vector<4xi32>
  %base = vector.addi %lv8, %four : vector<4xi32>
  // level 7 is the only one with all three bits set: bump = 2 (L & L >> 1 & L >> 2 & 1)
  %one7 = vector.splat %c1_i32 : vector<4xi32>
  %two7 = vector.splat %c2_i32 : vector<4xi32>
  %l1 = vector.shrui %lv, %one7 : vector<4xi32>
  %l2 = vector.shrui %lv, %two7 : vector<4xi32>
  %a01 = vector.andi %lv, %l1 : vector<4xi32>
  %a012 = vector.andi %a01, %l2 : vector<4xi32>
  %is7 = vector.andi %a012, %one7 : vector<4xi32>
  %bump = vector.shli %is7, %one7 : vector<4xi32>
  %mag = vector.addi %base, %bump : vector<4xi32>
  %sbase = scalar.muli %half, %c4_i32 : i32
  %c1v = scalar.constant 1 : i32
  %c2v = scalar.constant 2 : i32
  %c3v = scalar.constant 3 : i32
  %sh0 = vector.from_elements %s0, %c1v, %c2v, %c3v : vector<4xi32>
  %sbv = vector.splat %sbase : vector<4xi32>
  %sh = vector.addi %sh0, %sbv : vector<4xi32>
  %sv = vector.splat %signs8 : vector<4xi32>
  %sb0 = vector.shrui %sv, %sh : vector<4xi32>
  %one = vector.splat %c1_i32 : vector<4xi32>
  %sb = vector.andi %sb0, %one : vector<4xi32>
  %sb2 = vector.shli %sb, %one : vector<4xi32>
  %sgn = vector.subi %one, %sb2 : vector<4xi32>
  %signed = vector.muli %mag, %sgn : vector<4xi32>
  %f = vector.sitofp %signed : vector<4xi32> to vector<4xf32>
  %scv = vector.splat %scale : vector<4xf32>
  %result = vector.mulf %f, %scv : vector<4xf32>
  func.return %result : vector<4xf32>
}

// IQ3_XXS (98 bytes: d, qs[64] grid indices, 8 x u32 scales and signs; dequantize_row_iq3_xxs). Group g
// has grid indices qs[8 g .. 8 g + 7] (bytes 2 + 8 g ..) and aux = u32 at byte 66 + 4 g: four 7-bit sign
// groups (slot l at bit 7 l) and a 4-bit scale on top: value = d * (0.5 + (aux >> 28)) * 0.5 * grid * sign.
// Packet p covers slot p / 2, grid entry 2 (p / 2) + p % 2 (its four values, sign bits 4 (p % 2) ..).
func.def inline @ggml_iq3xxs_f32_vector4(%weight: buffer, %row_byte_base: offset, %iq3_block: index, %iq3_group: index, %packet: index) -> (vector<4xf32>) {
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c2 = index.constant 2 : index
  %c8 = index.constant 8 : index
  %c33 = index.constant 33 : index
  %c7_i32 = scalar.constant 7 : i32
  %c16_i32 = scalar.constant 16 : i32
  %c28_i32 = scalar.constant 28 : i32
  %c127_i32 = scalar.constant 127 : i32
  %c05_f32 = scalar.constant 0.5 : f32
  %block_bytes = index.constant 98 : offset
  %block_byte_add = index.scale %iq3_block, %block_bytes : index, offset -> offset
  %block_byte_base = index.add %row_byte_base, %block_byte_add : offset
  %hv = buffer.view %weight[%block_byte_base] : buffer -> view<49xf16>
  %wv = buffer.view %weight[%block_byte_base] : buffer -> view<49xi16>
  %bv = buffer.view %weight[%block_byte_base] : buffer -> view<98xi8>
  %g = index.assume %iq3_group [range(%iq3_group, 0, 7)] : index
  %p = index.assume %packet [range(%packet, 0, 7)] : index
  %slot = index.div %p, %c2 : index
  %half = index.rem %p, %c2 : index
  %g8 = index.mul %g, %c8 : index
  %idx0 = index.add %g8, %c2 : index
  %idx_at = index.add %idx0, %p : index
  %g2 = index.add %g, %g : index
  %aux_w0 = index.add %c33, %g2 : index
  %aux_w1 = index.add %aux_w0, %c1 : index
  %d_f16 = view.load %hv[%c0] : view<49xf16> -> f16
  %d = scalar.extf %d_f16 : f16 to f32
  %gi_i8 = view.load %bv[%idx_at] : view<98xi8> -> i8
  %gi = scalar.extui %gi_i8 : i8 to i32
  %w0_i16 = view.load %wv[%aux_w0] : view<49xi16> -> i16
  %w1_i16 = view.load %wv[%aux_w1] : view<49xi16> -> i16
  %w0 = scalar.extui %w0_i16 : i16 to i32
  %w1 = scalar.extui %w1_i16 : i16 to i32
  %w1s = scalar.shli %w1, %c16_i32 : i32
  %aux = scalar.ori %w0, %w1s : i32
  %sc4 = scalar.shrui %aux, %c28_i32 : i32
  %sc_f = scalar.uitofp %sc4 : i32 to f32
  %sc_plus = scalar.addf %sc_f, %c05_f32 : f32
  %ds = scalar.mulf %d, %sc_plus : f32
  %scale = scalar.mulf %ds, %c05_f32 : f32
  %slot_i32 = index.cast %slot : index to i32
  %sshift = scalar.muli %slot_i32, %c7_i32 : i32
  %sgrp = scalar.shrui %aux, %sshift : i32
  %signs7 = scalar.andi %sgrp, %c127_i32 : i32
  %signs8 = func.call @ggml_iq2_signs8(%signs7) : (i32) -> (i32)
  %code = func.call @ggml_iq3xxs_grid_code_i32(%gi) : (i32) -> (i32)
  %half_i32 = index.cast %half : index to i32
  %result = func.call @ggml_iq3xxs_code_vector4(%code, %half_i32, %signs8, %scale) : (i32, i32, i32, f32) -> (vector<4xf32>)
  func.return %result : vector<4xf32>
}

func.def inline @ggml_iq3xxs_f16_vector4(%weight: buffer, %row_byte_base: offset, %iq3_block: index, %iq3_group: index, %packet: index) -> (vector<4xf16>) {
  %values_f32 = func.call @ggml_iq3xxs_f32_vector4(%weight, %row_byte_base, %iq3_block, %iq3_group, %packet) : (buffer, offset, index, index, index) -> (vector<4xf32>)
  %values = vector.fptrunc %values_f32 : vector<4xf32> to vector<4xf16>
  func.return %values : vector<4xf16>
}
'''


def fill(fname, codes):
    per = len(codes) // 4
    o = [f'// Stages the IQ3_XXS grid, one 12-bit code per word; subgroup %chunk (0..3) writes words {per} %chunk .. +{per - 1}.',
         f'func.def inline @{fname}(%grid: buffer, %chunk: index) {{',
         '  %zero_offset = index.constant 0 : offset',
         '  %gv = buffer.view %grid[%zero_offset] : buffer -> view<512xi32>']
    for c in range(4):
        o += [f'  %k{c} = index.constant {c} : index', f'  %is{c} = index.cmp eq, %chunk, %k{c} : index', f'  scf.if %is{c} {{']
        for q in range(per // 4):
            e = per * c + 4 * q
            for i in range(4):
                o.append(f'    %v{c}_{q}_{i} = scalar.constant {codes[e + i]} : i32')
            o.append(f'    %w{c}_{q} = vector.from_elements %v{c}_{q}_0, %v{c}_{q}_1, %v{c}_{q}_2, %v{c}_{q}_3 : vector<4xi32>')
            o.append(f'    %o{c}_{q} = index.constant {e} : index')
            o.append(f'    vector.store %w{c}_{q}, %gv[%o{c}_{q}] : vector<4xi32>, view<512xi32>')
        o.append('  }')
    o += ['  func.return', '}', '']
    return '\n'.join(o)


kquant_common = '''
// IQ3_XXS: a 256-value block is 8 groups of 32, each 4 slots of 8 values (two grid entries of 4); lane l16
// owns group l16 / 2, slots 2 (l16 % 2) and +1: values 32 (l16 / 2) + 16 (l16 % 2) .. +15, one scale.
// The grid (12-bit codes) is staged in workgroup memory like IQ2's; signs are ksigns_iq2xs (7 bits + parity).
func.def inline @ggml_kquant_iq3xxs_slot_values(%code_a: i32, %code_b: i32, %signs7: i32) -> (vector<8xf32>) {
  %c1_i32 = scalar.constant 1 : i32
  %c2_i32 = scalar.constant 2 : i32
  %c4_i32 = scalar.constant 4 : i32
  %c7_i32 = scalar.constant 7 : i32
  %c12_i32 = scalar.constant 12 : i32
  %p4 = scalar.shrui %signs7, %c4_i32 : i32
  %x4 = scalar.xori %signs7, %p4 : i32
  %p2 = scalar.shrui %x4, %c2_i32 : i32
  %x2 = scalar.xori %x4, %p2 : i32
  %p1 = scalar.shrui %x2, %c1_i32 : i32
  %x1 = scalar.xori %x2, %p1 : i32
  %parity = scalar.andi %x1, %c1_i32 : i32
  %high = scalar.shli %parity, %c7_i32 : i32
  %signs8 = scalar.ori %signs7, %high : i32
  %b_hi = scalar.shli %code_b, %c12_i32 : i32
  %both = scalar.ori %code_a, %b_hi : i32
  %s0 = scalar.constant 0 : i32
  %s3 = scalar.constant 3 : i32
  %s5 = scalar.constant 5 : i32
  %s6 = scalar.constant 6 : i32
  %s8 = scalar.constant 8 : i32
  %s9 = scalar.constant 9 : i32
  %s12 = scalar.constant 12 : i32
  %s15 = scalar.constant 15 : i32
  %s18 = scalar.constant 18 : i32
  %s21 = scalar.constant 21 : i32
  %shift3 = vector.from_elements %s0, %s3, %s6, %s9, %s12, %s15, %s18, %s21 : vector<8xi32>
  %shift1 = vector.from_elements %s0, %c1_i32, %c2_i32, %s3, %c4_i32, %s5, %s6, %c7_i32 : vector<8xi32>
  %cv = vector.splat %both : vector<8xi32>
  %lvs = vector.shrui %cv, %shift3 : vector<8xi32>
  %seven = vector.splat %c7_i32 : vector<8xi32>
  %lv = vector.andi %lvs, %seven : vector<8xi32>
  %eight = vector.splat %s8 : vector<8xi32>
  %four = vector.splat %c4_i32 : vector<8xi32>
  %lv8 = vector.muli %lv, %eight : vector<8xi32>
  %base = vector.addi %lv8, %four : vector<8xi32>
  // level 7 is the only one with all three bits set: bump = 2 (L & L >> 1 & L >> 2 & 1)
  %one7 = vector.splat %c1_i32 : vector<8xi32>
  %two7 = vector.splat %c2_i32 : vector<8xi32>
  %l1 = vector.shrui %lv, %one7 : vector<8xi32>
  %l2 = vector.shrui %lv, %two7 : vector<8xi32>
  %a01 = vector.andi %lv, %l1 : vector<8xi32>
  %a012 = vector.andi %a01, %l2 : vector<8xi32>
  %is7 = vector.andi %a012, %one7 : vector<8xi32>
  %bump = vector.shli %is7, %one7 : vector<8xi32>
  %mag = vector.addi %base, %bump : vector<8xi32>
  %sv = vector.splat %signs8 : vector<8xi32>
  %sb0 = vector.shrui %sv, %shift1 : vector<8xi32>
  %one = vector.splat %c1_i32 : vector<8xi32>
  %sb = vector.andi %sb0, %one : vector<8xi32>
  %sb2 = vector.shli %sb, %one : vector<8xi32>
  %sgn = vector.subi %one, %sb2 : vector<8xi32>
  %v = vector.muli %mag, %sgn : vector<8xi32>
  %vf = vector.sitofp %v : vector<8xi32> to vector<8xf32>
  func.return %vf : vector<8xf32>
}

// IQ3_XXS (98 bytes): lane slots s = 2h, 2h + 1 of group g use grid indices at bytes 2 + 8 g + 2 s (+1)
// and sign group s of aux = u32 at byte 66 + 4 g, scale d * (0.5 + (aux >> 28)) * 0.5.
func.def inline @ggml_kquant_iq3xxs_lane_parts(%grid: buffer, %weight: buffer, %row_base: offset, %block: index, %lane16: index) -> (vector<16xf32>, f32, index) {
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c2 = index.constant 2 : index
  %c3 = index.constant 3 : index
  %c4 = index.constant 4 : index
  %c8 = index.constant 8 : index
  %c16 = index.constant 16 : index
  %c32 = index.constant 32 : index
  %c33 = index.constant 33 : index
  %c7_i32 = scalar.constant 7 : i32
  %c16_i32 = scalar.constant 16 : i32
  %c28_i32 = scalar.constant 28 : i32
  %c127_i32 = scalar.constant 127 : i32
  %c05 = scalar.constant 0.5 : f32
  %block_bytes = index.constant 98 : offset
  %zero_offset = index.constant 0 : offset
  %l = index.assume %lane16 [range(%lane16, 0, 15)] : index
  %g = index.div %l, %c2 : index
  %h = index.rem %l, %c2 : index
  %s0 = index.mul %h, %c2 : index
  %s1 = index.add %s0, %c1 : index
  %block_add = index.scale %block, %block_bytes : index, offset -> offset
  %block_base = index.add %row_base, %block_add : offset
  %hv = buffer.view %weight[%block_base] : buffer -> view<49xf16>
  %wv = buffer.view %weight[%block_base] : buffer -> view<49xi16>
  %bv = buffer.view %weight[%block_base] : buffer -> view<98xi8>
  %gv = buffer.view %grid[%zero_offset] : buffer -> view<512xi32>
  %d_f16 = view.load %hv[%c0] : view<49xf16> -> f16
  %d = scalar.extf %d_f16 : f16 to f32
  %g8 = index.mul %g, %c8 : index
  %i0 = index.add %g8, %c2 : index
  %h4 = index.mul %h, %c4 : index
  %ia = index.add %i0, %h4 : index
  %ib = index.add %ia, %c1 : index
  %ic = index.add %ia, %c2 : index
  %id = index.add %ia, %c3 : index
  %ia_i8 = view.load %bv[%ia] : view<98xi8> -> i8
  %ib_i8 = view.load %bv[%ib] : view<98xi8> -> i8
  %ic_i8 = view.load %bv[%ic] : view<98xi8> -> i8
  %id_i8 = view.load %bv[%id] : view<98xi8> -> i8
  %ga = scalar.extui %ia_i8 : i8 to i32
  %gb = scalar.extui %ib_i8 : i8 to i32
  %gc = scalar.extui %ic_i8 : i8 to i32
  %gd = scalar.extui %id_i8 : i8 to i32
  %ga_x = index.cast %ga : i32 to index
  %gb_x = index.cast %gb : i32 to index
  %gc_x = index.cast %gc : i32 to index
  %gd_x = index.cast %gd : i32 to index
  %ga_b = index.assume %ga_x [range(%ga_x, 0, 255)] : index
  %gb_b = index.assume %gb_x [range(%gb_x, 0, 255)] : index
  %gc_b = index.assume %gc_x [range(%gc_x, 0, 255)] : index
  %gd_b = index.assume %gd_x [range(%gd_x, 0, 255)] : index
  %code_a = view.load %gv[%ga_b] : view<512xi32> -> i32
  %code_b = view.load %gv[%gb_b] : view<512xi32> -> i32
  %code_c = view.load %gv[%gc_b] : view<512xi32> -> i32
  %code_d = view.load %gv[%gd_b] : view<512xi32> -> i32
  %g2 = index.add %g, %g : index
  %w0_at = index.add %c33, %g2 : index
  %w1_at = index.add %w0_at, %c1 : index
  %w0_i16 = view.load %wv[%w0_at] : view<49xi16> -> i16
  %w1_i16 = view.load %wv[%w1_at] : view<49xi16> -> i16
  %w0 = scalar.extui %w0_i16 : i16 to i32
  %w1 = scalar.extui %w1_i16 : i16 to i32
  %w1s = scalar.shli %w1, %c16_i32 : i32
  %aux = scalar.ori %w0, %w1s : i32
  %sc4 = scalar.shrui %aux, %c28_i32 : i32
  %sc_f = scalar.uitofp %sc4 : i32 to f32
  %sc_p = scalar.addf %sc_f, %c05 : f32
  %ds = scalar.mulf %d, %sc_p : f32
  %scale = scalar.mulf %ds, %c05 : f32
  %s0_i32 = index.cast %s0 : index to i32
  %s1_i32 = index.cast %s1 : index to i32
  %sha = scalar.muli %s0_i32, %c7_i32 : i32
  %shb = scalar.muli %s1_i32, %c7_i32 : i32
  %sa0 = scalar.shrui %aux, %sha : i32
  %sb0 = scalar.shrui %aux, %shb : i32
  %sa = scalar.andi %sa0, %c127_i32 : i32
  %sb = scalar.andi %sb0, %c127_i32 : i32
  %va = func.call @ggml_kquant_iq3xxs_slot_values(%code_a, %code_b, %sa) : (i32, i32, i32) -> (vector<8xf32>)
  %vb = func.call @ggml_kquant_iq3xxs_slot_values(%code_c, %code_d, %sb) : (i32, i32, i32) -> (vector<8xf32>)
  %v = func.call @ggml_kquant_iq2_join16(%va, %vb) : (vector<8xf32>, vector<8xf32>) -> (vector<16xf32>)
  %g32 = index.mul %g, %c32 : index
  %h16 = index.mul %h, %c16 : index
  %p0 = index.add %g32, %h16 : index
  func.return %v, %scale, %p0 : vector<16xf32>, f32, index
}

func.def inline @ggml_kquant_iq3xxs_lane_dot(%grid: buffer, %weight: buffer, %input: buffer, %row_base: offset, %block: index, %lane16: index) -> (f32) {
  %zero_scalar = scalar.constant 0.0 : f32
  %x_block_bytes = index.constant 1024 : offset
  %v, %scale, %p = func.call @ggml_kquant_iq3xxs_lane_parts(%grid, %weight, %row_base, %block, %lane16) : (buffer, buffer, offset, index, index) -> (vector<16xf32>, f32, index)
  %x_base = index.scale %block, %x_block_bytes : index, offset -> offset
  %xv = buffer.view %input[%x_base] : buffer -> view<256xf32>
  %x = vector.load %xv[%p] : view<256xf32> -> vector<16xf32>
  %vx = vector.mulf %v, %x : vector<16xf32>
  %sum = vector.reduce<addf> %vx, %zero_scalar : vector<16xf32>, f32
  %result = scalar.mulf %scale, %sum : f32
  func.return %result : f32
}

func.def inline @ggml_kquant_iq3xxs_lane_weights(%grid: buffer, %weight: buffer, %row_base: offset, %block: index, %lane16: index) -> (vector<16xf32>, index, index, index, index) {
  %c4 = index.constant 4 : index
  %c8 = index.constant 8 : index
  %c12 = index.constant 12 : index
  %v, %scale, %p0 = func.call @ggml_kquant_iq3xxs_lane_parts(%grid, %weight, %row_base, %block, %lane16) : (buffer, buffer, offset, index, index) -> (vector<16xf32>, f32, index)
  %sv = vector.splat %scale : vector<16xf32>
  %w = vector.mulf %v, %sv : vector<16xf32>
  %p1 = index.add %p0, %c4 : index
  %p2 = index.add %p0, %c8 : index
  %p3 = index.add %p0, %c12 : index
  func.return %w, %p0, %p1, %p2, %p3 : vector<16xf32>, index, index, index, index
}
'''

if mode == "dequant":
    out = lookup('ggml_iq3xxs_grid_code_i32', 'iq3xxs_code', grid()) + dequant_common
elif mode == "kquant":
    out = fill('ggml_kquant_iq3xxs_grid_fill', grid()) + kquant_common
else:
    sys.exit(f"unknown mode {mode}")
sys.stdout.write(out)
