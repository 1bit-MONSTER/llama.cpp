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
# Generates the Loom IQ2_XXS / IQ2_XS grid-code lookups and decoders for motifs/dequant.loom from
# ggml-common.h (iq2xxs_grid, iq2xs_grid: MIT, the ggml authors). Every grid byte is 8, 25 or 43,
# so an entry is stored as a 16-bit code: 2 bits per value, level index 0/1/2.
# Usage: generate_iq2_dequant_loom.py ggml/src/ggml-common.h > out.loom
# The output sits verbatim in kernel-corpus/kernels/loom-libs/motifs/dequant.loom.
import re
import sys
src = open(sys.argv[1]).read()


def grid(name, n):
    m = re.search(r'GGML_TABLE_BEGIN\(uint64_t, ' + name + r', ' + str(n) + r'\)(.*?)GGML_TABLE_END', src, re.S)
    assert m, name
    v = [int(x, 16) for x in re.findall(r'0x[0-9a-fA-F]+', m.group(1))]
    assert len(v) == n
    lv = {8: 0, 25: 1, 43: 2}
    return [sum(lv[(e >> (8 * j)) & 255] << (2 * j) for j in range(8)) for e in v]


def lookup(fname, prefix, codes):
    n = len(codes) // 32
    o = [f'// {prefix}: {len(codes)} grid entries as 16-bit codes (2 bits per value: 0 = 8, 1 = 25, 2 = 43)',
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


common = '''
// ksigns_iq2xs[i] (ggml-common.h) is i with its parity as bit 7.
func.def inline @ggml_iq2_signs8(%signs7: i32) -> (i32) {
  %c1_i32 = scalar.constant 1 : i32
  %c2_i32 = scalar.constant 2 : i32
  %c4_i32 = scalar.constant 4 : i32
  %c7_i32 = scalar.constant 7 : i32
  %p4 = scalar.shrui %signs7, %c4_i32 : i32
  %x4 = scalar.xori %signs7, %p4 : i32
  %p2 = scalar.shrui %x4, %c2_i32 : i32
  %x2 = scalar.xori %x4, %p2 : i32
  %p1 = scalar.shrui %x2, %c1_i32 : i32
  %x1 = scalar.xori %x2, %p1 : i32
  %parity = scalar.andi %x1, %c1_i32 : i32
  %high = scalar.shli %parity, %c7_i32 : i32
  %signs8 = scalar.ori %signs7, %high : i32
  func.return %signs8 : i32
}

// value j (0..7) of a grid code: level (8, 25, 43) with sign bit j of %signs8, times %scale
func.def inline @ggml_iq2_code_value_f32(%code: i32, %j: i32, %signs8: i32, %scale: f32) -> (f32) {
  %c0_i32 = scalar.constant 0 : i32
  %c1_i32 = scalar.constant 1 : i32
  %c3_i32 = scalar.constant 3 : i32
  %l8 = scalar.constant 8.0 : f32
  %l25 = scalar.constant 25.0 : f32
  %l43 = scalar.constant 43.0 : f32
  %two_j = scalar.addi %j, %j : i32
  %shifted = scalar.shrui %code, %two_j : i32
  %level = scalar.andi %shifted, %c3_i32 : i32
  %is0 = scalar.cmpi eq, %level, %c0_i32 : i32
  %is1 = scalar.cmpi eq, %level, %c1_i32 : i32
  %v01 = scf.select %is1, %l25, %l43 : f32
  %v = scf.select %is0, %l8, %v01 : f32
  %bit = scalar.shli %c1_i32, %j : i32
  %sign_mask = scalar.andi %signs8, %bit : i32
  %negative = scalar.cmpi ne, %sign_mask, %c0_i32 : i32
  %neg_v = scalar.negf %v : f32
  %signed = scf.select %negative, %neg_v, %v : f32
  %result = scalar.mulf %signed, %scale : f32
  func.return %result : f32
}

// four values 4 (p % 2) .. +3 of grid slot p / 2
func.def inline @ggml_iq2_code_vector4(%code: i32, %half: i32, %signs8: i32, %scale: f32) -> (vector<4xf32>) {
  %c1_i32 = scalar.constant 1 : i32
  %c2_i32 = scalar.constant 2 : i32
  %c3_i32 = scalar.constant 3 : i32
  %c4_i32 = scalar.constant 4 : i32
  %j0 = scalar.muli %half, %c4_i32 : i32
  %j1 = scalar.addi %j0, %c1_i32 : i32
  %j2 = scalar.addi %j0, %c2_i32 : i32
  %j3 = scalar.addi %j0, %c3_i32 : i32
  %v0 = func.call @ggml_iq2_code_value_f32(%code, %j0, %signs8, %scale) : (i32, i32, i32, f32) -> (f32)
  %v1 = func.call @ggml_iq2_code_value_f32(%code, %j1, %signs8, %scale) : (i32, i32, i32, f32) -> (f32)
  %v2 = func.call @ggml_iq2_code_value_f32(%code, %j2, %signs8, %scale) : (i32, i32, i32, f32) -> (f32)
  %v3 = func.call @ggml_iq2_code_value_f32(%code, %j3, %signs8, %scale) : (i32, i32, i32, f32) -> (f32)
  %result = vector.from_elements %v0, %v1, %v2, %v3 : vector<4xf32>
  func.return %result : vector<4xf32>
}

// IQ2_XXS (66 bytes: d, qs[32] u16; dequantize_row_iq2_xxs). Group g (ib32) is 8 bytes at 2 + 8g:
// four grid indices, then a u32 with four 7-bit sign groups and a 4-bit scale on top:
// value = d * (0.5 + (aux >> 28)) * 0.25 * grid * sign. Packet p covers slot p / 2, values 4 (p % 2) ..
func.def inline @ggml_iq2xxs_f32_vector4(%weight: buffer, %row_byte_base: offset, %iq2_block: index, %iq2_group: index, %packet: index) -> (vector<4xf32>) {
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c2 = index.constant 2 : index
  %c3 = index.constant 3 : index
  %c4 = index.constant 4 : index
  %c7_i32 = scalar.constant 7 : i32
  %c16_i32 = scalar.constant 16 : i32
  %c28_i32 = scalar.constant 28 : i32
  %c127_i32 = scalar.constant 127 : i32
  %c65535_i32 = scalar.constant 65535 : i32
  %c05_f32 = scalar.constant 0.5 : f32
  %c025_f32 = scalar.constant 0.25 : f32
  %block_bytes = index.constant 66 : offset
  %block_byte_add = index.scale %iq2_block, %block_bytes : index, offset -> offset
  %block_byte_base = index.add %row_byte_base, %block_byte_add : offset
  %hv = buffer.view %weight[%block_byte_base] : buffer -> view<33xf16>
  %wv = buffer.view %weight[%block_byte_base] : buffer -> view<33xi16>
  %bv = buffer.view %weight[%block_byte_base] : buffer -> view<66xi8>
  %g = index.assume %iq2_group [range(%iq2_group, 0, 7)] : index
  %p = index.assume %packet [range(%packet, 0, 7)] : index
  %slot = index.div %p, %c2 : index
  %half = index.rem %p, %c2 : index
  %g8 = index.mul %g, %c4 : index
  %gw = index.mul %g, %c4 : index
  %idx_byte0 = index.add %g8, %g8 : index
  %idx_byte1 = index.add %idx_byte0, %c2 : index
  %idx_byte = index.add %idx_byte1, %slot : index
  %aux_w0_0 = index.add %gw, %c3 : index
  %aux_w1_0 = index.add %aux_w0_0, %c1 : index
  %d_f16 = view.load %hv[%c0] : view<33xf16> -> f16
  %d = scalar.extf %d_f16 : f16 to f32
  %gi_i8 = view.load %bv[%idx_byte] : view<66xi8> -> i8
  %gi = scalar.extui %gi_i8 : i8 to i32
  %w0_i16 = view.load %wv[%aux_w0_0] : view<33xi16> -> i16
  %w1_i16 = view.load %wv[%aux_w1_0] : view<33xi16> -> i16
  %w0 = scalar.extui %w0_i16 : i16 to i32
  %w1 = scalar.extui %w1_i16 : i16 to i32
  %w1s = scalar.shli %w1, %c16_i32 : i32
  %aux = scalar.ori %w0, %w1s : i32
  %sc4 = scalar.shrui %aux, %c28_i32 : i32
  %sc_f = scalar.uitofp %sc4 : i32 to f32
  %sc_plus = scalar.addf %sc_f, %c05_f32 : f32
  %ds = scalar.mulf %d, %sc_plus : f32
  %scale = scalar.mulf %ds, %c025_f32 : f32
  %slot_i32 = index.cast %slot : index to i32
  %sshift = scalar.muli %slot_i32, %c7_i32 : i32
  %sgrp0 = scalar.shrui %aux, %sshift : i32
  %signs7 = scalar.andi %sgrp0, %c127_i32 : i32
  %signs8 = func.call @ggml_iq2_signs8(%signs7) : (i32) -> (i32)
  %code = func.call @ggml_iq2xxs_grid_code_i32(%gi) : (i32) -> (i32)
  %half_i32 = index.cast %half : index to i32
  %result = func.call @ggml_iq2_code_vector4(%code, %half_i32, %signs8, %scale) : (i32, i32, i32, f32) -> (vector<4xf32>)
  func.return %result : vector<4xf32>
}

func.def inline @ggml_iq2xxs_f16_vector4(%weight: buffer, %row_byte_base: offset, %iq2_block: index, %iq2_group: index, %packet: index) -> (vector<4xf16>) {
  %values_f32 = func.call @ggml_iq2xxs_f32_vector4(%weight, %row_byte_base, %iq2_block, %iq2_group, %packet) : (buffer, offset, index, index, index) -> (vector<4xf32>)
  %values = vector.fptrunc %values_f32 : vector<4xf32> to vector<4xf16>
  func.return %values : vector<4xf16>
}

// IQ2_XS (74 bytes: d, qs[32] u16, scales[8]; dequantize_row_iq2_xs). Slot l of group g is
// q = qs[4g + l]: grid index q & 511, 7 sign bits q >> 9; the scale nibble (l / 2) of scales[g]:
// value = d * (0.5 + nibble) * 0.25 * grid * sign.
func.def inline @ggml_iq2xs_f32_vector4(%weight: buffer, %row_byte_base: offset, %iq2_block: index, %iq2_group: index, %packet: index) -> (vector<4xf32>) {
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c2 = index.constant 2 : index
  %c4 = index.constant 4 : index
  %c66 = index.constant 66 : index
  %c4_i32 = scalar.constant 4 : i32
  %c9_i32 = scalar.constant 9 : i32
  %c15_i32 = scalar.constant 15 : i32
  %c511_i32 = scalar.constant 511 : i32
  %c127_i32 = scalar.constant 127 : i32
  %c05_f32 = scalar.constant 0.5 : f32
  %c025_f32 = scalar.constant 0.25 : f32
  %block_bytes = index.constant 74 : offset
  %block_byte_add = index.scale %iq2_block, %block_bytes : index, offset -> offset
  %block_byte_base = index.add %row_byte_base, %block_byte_add : offset
  %hv = buffer.view %weight[%block_byte_base] : buffer -> view<37xf16>
  %wv = buffer.view %weight[%block_byte_base] : buffer -> view<37xi16>
  %bv = buffer.view %weight[%block_byte_base] : buffer -> view<74xi8>
  %g = index.assume %iq2_group [range(%iq2_group, 0, 7)] : index
  %p = index.assume %packet [range(%packet, 0, 7)] : index
  %slot = index.div %p, %c2 : index
  %half = index.rem %p, %c2 : index
  %g4 = index.mul %g, %c4 : index
  %q_at0 = index.add %g4, %slot : index
  %q_at = index.add %q_at0, %c1 : index
  %sc_at = index.add %c66, %g : index
  %d_f16 = view.load %hv[%c0] : view<37xf16> -> f16
  %d = scalar.extf %d_f16 : f16 to f32
  %q_i16 = view.load %wv[%q_at] : view<37xi16> -> i16
  %q = scalar.extui %q_i16 : i16 to i32
  %gi = scalar.andi %q, %c511_i32 : i32
  %signs7_0 = scalar.shrui %q, %c9_i32 : i32
  %signs7 = scalar.andi %signs7_0, %c127_i32 : i32
  %sc_i8 = view.load %bv[%sc_at] : view<74xi8> -> i8
  %sc = scalar.extui %sc_i8 : i8 to i32
  %nib_sel = index.div %slot, %c2 : index
  %nib_sel_i32 = index.cast %nib_sel : index to i32
  %nib_shift = scalar.muli %nib_sel_i32, %c4_i32 : i32
  %nib0 = scalar.shrui %sc, %nib_shift : i32
  %nib = scalar.andi %nib0, %c15_i32 : i32
  %nib_f = scalar.uitofp %nib : i32 to f32
  %nib_plus = scalar.addf %nib_f, %c05_f32 : f32
  %ds = scalar.mulf %d, %nib_plus : f32
  %scale = scalar.mulf %ds, %c025_f32 : f32
  %signs8 = func.call @ggml_iq2_signs8(%signs7) : (i32) -> (i32)
  %code = func.call @ggml_iq2xs_grid_code_i32(%gi) : (i32) -> (i32)
  %half_i32 = index.cast %half : index to i32
  %result = func.call @ggml_iq2_code_vector4(%code, %half_i32, %signs8, %scale) : (i32, i32, i32, f32) -> (vector<4xf32>)
  func.return %result : vector<4xf32>
}

func.def inline @ggml_iq2xs_f16_vector4(%weight: buffer, %row_byte_base: offset, %iq2_block: index, %iq2_group: index, %packet: index) -> (vector<4xf16>) {
  %values_f32 = func.call @ggml_iq2xs_f32_vector4(%weight, %row_byte_base, %iq2_block, %iq2_group, %packet) : (buffer, offset, index, index, index) -> (vector<4xf32>)
  %values = vector.fptrunc %values_f32 : vector<4xf32> to vector<4xf16>
  func.return %values : vector<4xf16>
}
'''
out = lookup('ggml_iq2xxs_grid_code_i32', 'iq2xxs_code', grid('iq2xxs_grid', 256)) + '\n' + \
    lookup('ggml_iq2xs_grid_code_i32', 'iq2xs_code', grid('iq2xs_grid', 512)) + common
sys.stdout.write(out)
