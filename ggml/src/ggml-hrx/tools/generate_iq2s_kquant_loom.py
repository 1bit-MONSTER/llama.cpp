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
# IQ2_S lane functions and grid fill for ops/kquant_decode_f32.loom, generated from ggml-common.h
# (iq2s_grid: MIT, the ggml authors). Every grid byte is 8, 25 or 43, so an entry (8 values) is a
# 16-bit code (2 bits per value, level 0/1/2); the 1024 codes are staged two per word (512 words, the
# kernels' 2 KiB grid buffer).
# Usage: generate_iq2s_kquant_loom.py ggml/src/ggml-common.h > out.loom
# The output sits verbatim in kernel-corpus/kernels/loom-libs/ops/kquant_decode_f32.loom.
import re
import sys

src = open(sys.argv[1]).read()


def grid():
    m = re.search(r'GGML_TABLE_BEGIN\(uint64_t, iq2s_grid, 1024\)(.*?)GGML_TABLE_END', src, re.S)
    assert m, "iq2s_grid"
    v = [int(x, 16) for x in re.findall(r'0x[0-9a-fA-F]+', m.group(1))]
    assert len(v) == 1024, len(v)
    lv = {8: 0, 25: 1, 43: 2}
    return [sum(lv[(e >> (8 * j)) & 255] << (2 * j) for j in range(8)) for e in v]


def fill(fname, codes):
    words = [codes[2 * w] | (codes[2 * w + 1] << 16) for w in range(len(codes) // 2)]
    words = [w - (1 << 32) if w >= 1 << 31 else w for w in words]  # i32 constants are signed
    per = len(words) // 4
    o = [f'// Stages the IQ2_S grid as {len(words)} words (two 16-bit codes each); subgroup %chunk (0..3) writes',
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


common = '''
// IQ2_S: lane l16 owns group l16 / 2, slots 2 (l16 % 2) and +1 (values 32 (l16 / 2) + 16 (l16 % 2) ..
// +15), which share scale nibble l16 % 2 of scales[g]. Slot s: grid index qs[4 g + s] | (qh[g] << (8 - 2 s)
// & 0x300), a full sign byte signs[4 g + s] (no parity bit, unlike IQ2_XXS / IQ2_XS).
func.def inline @ggml_kquant_iq2s_code(%grid: buffer, %index: i32) -> (i32) {
  %zero_offset = index.constant 0 : offset
  %gv = buffer.view %grid[%zero_offset] : buffer -> view<512xi32>
  %c1_i32 = scalar.constant 1 : i32
  %c4_i32 = scalar.constant 4 : i32
  %c65535_i32 = scalar.constant 65535 : i32
  %w_i32 = scalar.shrui %index, %c1_i32 : i32
  %w_x = index.cast %w_i32 : i32 to index
  %w_b = index.assume %w_x [range(%w_x, 0, 511)] : index
  %word = view.load %gv[%w_b] : view<512xi32> -> i32
  %odd = scalar.andi %index, %c1_i32 : i32
  %sh = scalar.shli %odd, %c4_i32 : i32
  %shifted = scalar.shrui %word, %sh : i32
  %code = scalar.andi %shifted, %c65535_i32 : i32
  func.return %code : i32
}

func.def inline @ggml_kquant_iq2s_slot_values(%code: i32, %signs8: i32) -> (vector<8xf32>) {
  %c1_i32 = scalar.constant 1 : i32
  %c7_i32 = scalar.constant 7 : i32
  %s0 = scalar.constant 0 : i32
  %s2 = scalar.constant 2 : i32
  %s3 = scalar.constant 3 : i32
  %s4 = scalar.constant 4 : i32
  %s5 = scalar.constant 5 : i32
  %s6 = scalar.constant 6 : i32
  %s8 = scalar.constant 8 : i32
  %s10 = scalar.constant 10 : i32
  %s12 = scalar.constant 12 : i32
  %s14 = scalar.constant 14 : i32
  %shift2 = vector.from_elements %s0, %s2, %s4, %s6, %s8, %s10, %s12, %s14 : vector<8xi32>
  %shift1 = vector.from_elements %s0, %c1_i32, %s2, %s3, %s4, %s5, %s6, %c7_i32 : vector<8xi32>
  %three = vector.splat %s3 : vector<8xi32>
  %one = vector.splat %c1_i32 : vector<8xi32>
  %c8v = vector.splat %s8 : vector<8xi32>
  %c17_i32 = scalar.constant 17 : i32
  %c17v = vector.splat %c17_i32 : vector<8xi32>
  %cv = vector.splat %code : vector<8xi32>
  %cs = vector.shrui %cv, %shift2 : vector<8xi32>
  %c = vector.andi %cs, %three : vector<8xi32>
  %c17 = vector.muli %c, %c17v : vector<8xi32>
  %chi = vector.shrui %c, %one : vector<8xi32>
  %lv0 = vector.addi %c17, %chi : vector<8xi32>
  %lv = vector.addi %lv0, %c8v : vector<8xi32>
  %sv = vector.splat %signs8 : vector<8xi32>
  %sb0 = vector.shrui %sv, %shift1 : vector<8xi32>
  %sb = vector.andi %sb0, %one : vector<8xi32>
  %sb2 = vector.shli %sb, %one : vector<8xi32>
  %sgn = vector.subi %one, %sb2 : vector<8xi32>
  %v = vector.muli %lv, %sgn : vector<8xi32>
  %vf = vector.sitofp %v : vector<8xi32> to vector<8xf32>
  func.return %vf : vector<8xf32>
}

// IQ2_S (82 bytes: d, qs[32] grid low bits, signs[32], qh[8], scales[8]): value = d * (0.5 + nibble) * 0.25
// * grid * sign.
func.def inline @ggml_kquant_iq2s_lane_parts(%grid: buffer, %weight: buffer, %row_base: offset, %block: index, %lane16: index) -> (vector<16xf32>, f32, index) {
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c2 = index.constant 2 : index
  %c4 = index.constant 4 : index
  %c16 = index.constant 16 : index
  %c32 = index.constant 32 : index
  %c34 = index.constant 34 : index
  %c66 = index.constant 66 : index
  %c74 = index.constant 74 : index
  %c2_i32 = scalar.constant 2 : i32
  %c4_i32 = scalar.constant 4 : i32
  %c8_i32 = scalar.constant 8 : i32
  %c15_i32 = scalar.constant 15 : i32
  %c768_i32 = scalar.constant 768 : i32
  %c05 = scalar.constant 0.5 : f32
  %c025 = scalar.constant 0.25 : f32
  %block_bytes = index.constant 82 : offset
  %l = index.assume %lane16 [range(%lane16, 0, 15)] : index
  %g = index.div %l, %c2 : index
  %h = index.rem %l, %c2 : index
  %sa = index.mul %h, %c2 : index
  %sb = index.add %sa, %c1 : index
  %block_add = index.scale %block, %block_bytes : index, offset -> offset
  %block_base = index.add %row_base, %block_add : offset
  %hv = buffer.view %weight[%block_base] : buffer -> view<41xf16>
  %bv = buffer.view %weight[%block_base] : buffer -> view<82xi8>
  %d_f16 = view.load %hv[%c0] : view<41xf16> -> f16
  %d = scalar.extf %d_f16 : f16 to f32
  %g4 = index.mul %g, %c4 : index
  %qs0 = index.add %g4, %c2 : index
  %qa_at = index.add %qs0, %sa : index
  %qb_at = index.add %qs0, %sb : index
  %sg0 = index.add %g4, %c34 : index
  %sa_at = index.add %sg0, %sa : index
  %sb_at = index.add %sg0, %sb : index
  %qh_at = index.add %c66, %g : index
  %sc_at = index.add %c74, %g : index
  %qa_i8 = view.load %bv[%qa_at] : view<82xi8> -> i8
  %qb_i8 = view.load %bv[%qb_at] : view<82xi8> -> i8
  %sga_i8 = view.load %bv[%sa_at] : view<82xi8> -> i8
  %sgb_i8 = view.load %bv[%sb_at] : view<82xi8> -> i8
  %qh_i8 = view.load %bv[%qh_at] : view<82xi8> -> i8
  %sc_i8 = view.load %bv[%sc_at] : view<82xi8> -> i8
  %qa = scalar.extui %qa_i8 : i8 to i32
  %qb = scalar.extui %qb_i8 : i8 to i32
  %sga = scalar.extui %sga_i8 : i8 to i32
  %sgb = scalar.extui %sgb_i8 : i8 to i32
  %qh = scalar.extui %qh_i8 : i8 to i32
  %sc = scalar.extui %sc_i8 : i8 to i32
  %sa_i32 = index.cast %sa : index to i32
  %sb_i32 = index.cast %sb : index to i32
  %sa2 = scalar.muli %sa_i32, %c2_i32 : i32
  %sb2 = scalar.muli %sb_i32, %c2_i32 : i32
  %sha = scalar.subi %c8_i32, %sa2 : i32
  %shb = scalar.subi %c8_i32, %sb2 : i32
  %ha0 = scalar.shli %qh, %sha : i32
  %hb0 = scalar.shli %qh, %shb : i32
  %ha = scalar.andi %ha0, %c768_i32 : i32
  %hb = scalar.andi %hb0, %c768_i32 : i32
  %ia = scalar.ori %qa, %ha : i32
  %ib = scalar.ori %qb, %hb : i32
  %code_a = func.call @ggml_kquant_iq2s_code(%grid, %ia) : (buffer, i32) -> (i32)
  %code_b = func.call @ggml_kquant_iq2s_code(%grid, %ib) : (buffer, i32) -> (i32)
  %h_i32 = index.cast %h : index to i32
  %nsh = scalar.muli %h_i32, %c4_i32 : i32
  %nib0 = scalar.shrui %sc, %nsh : i32
  %nib = scalar.andi %nib0, %c15_i32 : i32
  %nib_f = scalar.uitofp %nib : i32 to f32
  %nib_p = scalar.addf %nib_f, %c05 : f32
  %ds = scalar.mulf %d, %nib_p : f32
  %scale = scalar.mulf %ds, %c025 : f32
  %va = func.call @ggml_kquant_iq2s_slot_values(%code_a, %sga) : (i32, i32) -> (vector<8xf32>)
  %vb = func.call @ggml_kquant_iq2s_slot_values(%code_b, %sgb) : (i32, i32) -> (vector<8xf32>)
  %v = func.call @ggml_kquant_iq2_join16(%va, %vb) : (vector<8xf32>, vector<8xf32>) -> (vector<16xf32>)
  %g32 = index.mul %g, %c32 : index
  %h16 = index.mul %h, %c16 : index
  %p0 = index.add %g32, %h16 : index
  func.return %v, %scale, %p0 : vector<16xf32>, f32, index
}

func.def inline @ggml_kquant_iq2s_lane_dot(%grid: buffer, %weight: buffer, %input: buffer, %row_base: offset, %block: index, %lane16: index) -> (f32) {
  %zero_scalar = scalar.constant 0.0 : f32
  %x_block_bytes = index.constant 1024 : offset
  %v, %scale, %p = func.call @ggml_kquant_iq2s_lane_parts(%grid, %weight, %row_base, %block, %lane16) : (buffer, buffer, offset, index, index) -> (vector<16xf32>, f32, index)
  %x_base = index.scale %block, %x_block_bytes : index, offset -> offset
  %xv = buffer.view %input[%x_base] : buffer -> view<256xf32>
  %x = vector.load %xv[%p] : view<256xf32> -> vector<16xf32>
  %vx = vector.mulf %v, %x : vector<16xf32>
  %sum = vector.reduce<addf> %vx, %zero_scalar : vector<16xf32>, f32
  %result = scalar.mulf %scale, %sum : f32
  func.return %result : f32
}

func.def inline @ggml_kquant_iq2s_lane_weights(%grid: buffer, %weight: buffer, %row_base: offset, %block: index, %lane16: index) -> (vector<16xf32>, index, index, index, index) {
  %c4 = index.constant 4 : index
  %c8 = index.constant 8 : index
  %c12 = index.constant 12 : index
  %v, %scale, %p0 = func.call @ggml_kquant_iq2s_lane_parts(%grid, %weight, %row_base, %block, %lane16) : (buffer, buffer, offset, index, index) -> (vector<16xf32>, f32, index)
  %sv = vector.splat %scale : vector<16xf32>
  %w = vector.mulf %v, %sv : vector<16xf32>
  %p1 = index.add %p0, %c4 : index
  %p2 = index.add %p0, %c8 : index
  %p3 = index.add %p0, %c12 : index
  func.return %w, %p0, %p1, %p2, %p3 : vector<16xf32>, index, index, index, index
}
'''

sys.stdout.write(fill('ggml_kquant_iq2s_grid_fill', grid()) + common)
