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
# IQ2_XXS / IQ2_XS lane functions and grid fills for ops/kquant_decode_f32.loom, generated from
# ggml-common.h (iq2xxs_grid, iq2xs_grid: MIT, the ggml authors) as 16-bit codes (2 bits per value).
# Usage: generate_iq2_kquant_loom.py ggml/src/ggml-common.h > out.loom
# The output sits verbatim in kernel-corpus/kernels/loom-libs/ops/kquant_decode_f32.loom.
import re, sys
src = open(sys.argv[1]).read()
def grid(name, n):
    m = re.search(r'GGML_TABLE_BEGIN\(uint64_t, ' + name + r', ' + str(n) + r'\)(.*?)GGML_TABLE_END', src, re.S)
    v = [int(x, 16) for x in re.findall(r'0x[0-9a-fA-F]+', m.group(1))]
    lv = {8: 0, 25: 1, 43: 2}
    return [sum(lv[(e >> (8 * j)) & 255] << (2 * j) for j in range(8)) for e in v]

def fill(fname, codes):
    o = [f'func.def inline @{fname}(%grid: buffer, %chunk: index) {{',
         '  %zero_offset = index.constant 0 : offset',
         '  %gv = buffer.view %grid[%zero_offset] : buffer -> view<512xi32>']
    for c in range(len(codes) // 128):
        o += [f'  %k{c} = index.constant {c} : index', f'  %is{c} = index.cmp eq, %chunk, %k{c} : index', f'  scf.if %is{c} {{']
        for q in range(32):
            e = 128 * c + 4 * q
            for i in range(4):
                o.append(f'    %v{c}_{q}_{i} = scalar.constant {codes[e + i]} : i32')
            o.append(f'    %w{c}_{q} = vector.from_elements %v{c}_{q}_0, %v{c}_{q}_1, %v{c}_{q}_2, %v{c}_{q}_3 : vector<4xi32>')
            o.append(f'    %o{c}_{q} = index.constant {e} : index')
            o.append(f'    vector.store %w{c}_{q}, %gv[%o{c}_{q}] : vector<4xi32>, view<512xi32>')
        o.append('  }')
    o += ['  func.return', '}', '']
    return '\n'.join(o)

common = '''
// IQ2_XXS / IQ2_XS: a 256-value block is 8 groups of 32, each 4 grid slots of 8 values; lane l16
// owns group l16 / 2, slots 2 (l16 % 2) and +1: values 32 (l16 / 2) + 16 (l16 % 2) .. +15, one
// scale. The grid (16-bit codes, 2 bits per value: level 8 + 17c + (c >> 1) = 8, 25, 43) is
// staged in workgroup memory like IQ3_S's; the sign byte is ksigns_iq2xs = 7 bits + parity.
func.def inline @ggml_kquant_iq2_slot_values(%code: i32, %signs7: i32) -> (vector<8xf32>) {
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
  %s0 = scalar.constant 0 : i32
  %s2 = scalar.constant 2 : i32
  %s4 = scalar.constant 4 : i32
  %s6 = scalar.constant 6 : i32
  %s8 = scalar.constant 8 : i32
  %s10 = scalar.constant 10 : i32
  %s12 = scalar.constant 12 : i32
  %s14 = scalar.constant 14 : i32
  %s3 = scalar.constant 3 : i32
  %s5 = scalar.constant 5 : i32
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

func.def inline @ggml_kquant_iq2_join16(%a: vector<8xf32>, %b: vector<8xf32>) -> (vector<16xf32>) {
  %a0 = vector.extract %a[0] : vector<8xf32> -> f32
  %a1 = vector.extract %a[1] : vector<8xf32> -> f32
  %a2 = vector.extract %a[2] : vector<8xf32> -> f32
  %a3 = vector.extract %a[3] : vector<8xf32> -> f32
  %a4 = vector.extract %a[4] : vector<8xf32> -> f32
  %a5 = vector.extract %a[5] : vector<8xf32> -> f32
  %a6 = vector.extract %a[6] : vector<8xf32> -> f32
  %a7 = vector.extract %a[7] : vector<8xf32> -> f32
  %b0 = vector.extract %b[0] : vector<8xf32> -> f32
  %b1 = vector.extract %b[1] : vector<8xf32> -> f32
  %b2 = vector.extract %b[2] : vector<8xf32> -> f32
  %b3 = vector.extract %b[3] : vector<8xf32> -> f32
  %b4 = vector.extract %b[4] : vector<8xf32> -> f32
  %b5 = vector.extract %b[5] : vector<8xf32> -> f32
  %b6 = vector.extract %b[6] : vector<8xf32> -> f32
  %b7 = vector.extract %b[7] : vector<8xf32> -> f32
  %r = vector.from_elements %a0, %a1, %a2, %a3, %a4, %a5, %a6, %a7, %b0, %b1, %b2, %b3, %b4, %b5, %b6, %b7 : vector<16xf32>
  func.return %r : vector<16xf32>
}

// IQ2_XXS (66 bytes: d, qs[32] u16): group g = 8 bytes at 2 + 8g: slot indices (bytes 0..3),
// then u32 aux: four 7-bit sign groups, scale nibble in bits 28..31 (d * (0.5 + s) * 0.25).
func.def inline @ggml_kquant_iq2xxs_lane_parts(%grid: buffer, %weight: buffer, %row_base: offset, %block: index, %lane16: index) -> (vector<16xf32>, f32, index) {
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c2 = index.constant 2 : index
  %c3 = index.constant 3 : index
  %c4 = index.constant 4 : index
  %c8 = index.constant 8 : index
  %c16 = index.constant 16 : index
  %c32 = index.constant 32 : index
  %c7_i32 = scalar.constant 7 : i32
  %c16_i32 = scalar.constant 16 : i32
  %c28_i32 = scalar.constant 28 : i32
  %c127_i32 = scalar.constant 127 : i32
  %c05 = scalar.constant 0.5 : f32
  %c025 = scalar.constant 0.25 : f32
  %block_bytes = index.constant 66 : offset
  %zero_offset = index.constant 0 : offset
  %l = index.assume %lane16 [range(%lane16, 0, 15)] : index
  %g = index.div %l, %c2 : index
  %h = index.rem %l, %c2 : index
  %s0 = index.mul %h, %c2 : index
  %s1 = index.add %s0, %c1 : index
  %block_add = index.scale %block, %block_bytes : index, offset -> offset
  %block_base = index.add %row_base, %block_add : offset
  %hv = buffer.view %weight[%block_base] : buffer -> view<33xf16>
  %wv = buffer.view %weight[%block_base] : buffer -> view<33xi16>
  %bv = buffer.view %weight[%block_base] : buffer -> view<66xi8>
  %gv = buffer.view %grid[%zero_offset] : buffer -> view<512xi32>
  %d_f16 = view.load %hv[%c0] : view<33xf16> -> f16
  %d = scalar.extf %d_f16 : f16 to f32
  %g8 = index.mul %g, %c8 : index
  %ib0 = index.add %g8, %c2 : index
  %ia = index.add %ib0, %s0 : index
  %ib = index.add %ib0, %s1 : index
  %ia_i8 = view.load %bv[%ia] : view<66xi8> -> i8
  %ib_i8 = view.load %bv[%ib] : view<66xi8> -> i8
  %ga = scalar.extui %ia_i8 : i8 to i32
  %gb = scalar.extui %ib_i8 : i8 to i32
  %ga_x = index.cast %ga : i32 to index
  %gb_x = index.cast %gb : i32 to index
  %ga_b = index.assume %ga_x [range(%ga_x, 0, 255)] : index
  %gb_b = index.assume %gb_x [range(%gb_x, 0, 255)] : index
  %code_a = view.load %gv[%ga_b] : view<512xi32> -> i32
  %code_b = view.load %gv[%gb_b] : view<512xi32> -> i32
  %g4 = index.mul %g, %c4 : index
  %w0_at = index.add %g4, %c3 : index
  %w1_at = index.add %g4, %c4 : index
  %w0_i16 = view.load %wv[%w0_at] : view<33xi16> -> i16
  %w1_i16 = view.load %wv[%w1_at] : view<33xi16> -> i16
  %w0 = scalar.extui %w0_i16 : i16 to i32
  %w1 = scalar.extui %w1_i16 : i16 to i32
  %w1s = scalar.shli %w1, %c16_i32 : i32
  %aux = scalar.ori %w0, %w1s : i32
  %sc4 = scalar.shrui %aux, %c28_i32 : i32
  %sc_f = scalar.uitofp %sc4 : i32 to f32
  %sc_p = scalar.addf %sc_f, %c05 : f32
  %ds = scalar.mulf %d, %sc_p : f32
  %scale = scalar.mulf %ds, %c025 : f32
  %s0_i32 = index.cast %s0 : index to i32
  %s1_i32 = index.cast %s1 : index to i32
  %sha = scalar.muli %s0_i32, %c7_i32 : i32
  %shb = scalar.muli %s1_i32, %c7_i32 : i32
  %sa0 = scalar.shrui %aux, %sha : i32
  %sb0 = scalar.shrui %aux, %shb : i32
  %sa = scalar.andi %sa0, %c127_i32 : i32
  %sb = scalar.andi %sb0, %c127_i32 : i32
  %va = func.call @ggml_kquant_iq2_slot_values(%code_a, %sa) : (i32, i32) -> (vector<8xf32>)
  %vb = func.call @ggml_kquant_iq2_slot_values(%code_b, %sb) : (i32, i32) -> (vector<8xf32>)
  %v = func.call @ggml_kquant_iq2_join16(%va, %vb) : (vector<8xf32>, vector<8xf32>) -> (vector<16xf32>)
  %g32 = index.mul %g, %c32 : index
  %h16 = index.mul %h, %c16 : index
  %p0 = index.add %g32, %h16 : index
  func.return %v, %scale, %p0 : vector<16xf32>, f32, index
}

// IQ2_XS (74 bytes: d, qs[32] u16, scales[8]): slot q = qs[4g + s]: grid index q & 511, 7 sign
// bits q >> 9; scale nibble h of scales[g] (slots 2h, 2h + 1).
func.def inline @ggml_kquant_iq2xs_lane_parts(%grid: buffer, %weight: buffer, %row_base: offset, %block: index, %lane16: index) -> (vector<16xf32>, f32, index) {
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c2 = index.constant 2 : index
  %c4 = index.constant 4 : index
  %c16 = index.constant 16 : index
  %c32 = index.constant 32 : index
  %c66 = index.constant 66 : index
  %c4_i32 = scalar.constant 4 : i32
  %c9_i32 = scalar.constant 9 : i32
  %c15_i32 = scalar.constant 15 : i32
  %c127_i32 = scalar.constant 127 : i32
  %c511_i32 = scalar.constant 511 : i32
  %c05 = scalar.constant 0.5 : f32
  %c025 = scalar.constant 0.25 : f32
  %block_bytes = index.constant 74 : offset
  %zero_offset = index.constant 0 : offset
  %l = index.assume %lane16 [range(%lane16, 0, 15)] : index
  %g = index.div %l, %c2 : index
  %h = index.rem %l, %c2 : index
  %s0 = index.mul %h, %c2 : index
  %block_add = index.scale %block, %block_bytes : index, offset -> offset
  %block_base = index.add %row_base, %block_add : offset
  %hv = buffer.view %weight[%block_base] : buffer -> view<37xf16>
  %wv = buffer.view %weight[%block_base] : buffer -> view<37xi16>
  %bv = buffer.view %weight[%block_base] : buffer -> view<74xi8>
  %gv = buffer.view %grid[%zero_offset] : buffer -> view<512xi32>
  %d_f16 = view.load %hv[%c0] : view<37xf16> -> f16
  %d = scalar.extf %d_f16 : f16 to f32
  %g4 = index.mul %g, %c4 : index
  %qa_at0 = index.add %g4, %s0 : index
  %qa_at = index.add %qa_at0, %c1 : index
  %qb_at = index.add %qa_at, %c1 : index
  %qa_i16 = view.load %wv[%qa_at] : view<37xi16> -> i16
  %qb_i16 = view.load %wv[%qb_at] : view<37xi16> -> i16
  %qa = scalar.extui %qa_i16 : i16 to i32
  %qb = scalar.extui %qb_i16 : i16 to i32
  %ga = scalar.andi %qa, %c511_i32 : i32
  %gb = scalar.andi %qb, %c511_i32 : i32
  %sa0 = scalar.shrui %qa, %c9_i32 : i32
  %sb0 = scalar.shrui %qb, %c9_i32 : i32
  %sa = scalar.andi %sa0, %c127_i32 : i32
  %sb = scalar.andi %sb0, %c127_i32 : i32
  %ga_x = index.cast %ga : i32 to index
  %gb_x = index.cast %gb : i32 to index
  %ga_b = index.assume %ga_x [range(%ga_x, 0, 511)] : index
  %gb_b = index.assume %gb_x [range(%gb_x, 0, 511)] : index
  %code_a = view.load %gv[%ga_b] : view<512xi32> -> i32
  %code_b = view.load %gv[%gb_b] : view<512xi32> -> i32
  %sc_at = index.add %c66, %g : index
  %sc_i8 = view.load %bv[%sc_at] : view<74xi8> -> i8
  %sc = scalar.extui %sc_i8 : i8 to i32
  %h_i32 = index.cast %h : index to i32
  %nsh = scalar.muli %h_i32, %c4_i32 : i32
  %nib0 = scalar.shrui %sc, %nsh : i32
  %nib = scalar.andi %nib0, %c15_i32 : i32
  %nib_f = scalar.uitofp %nib : i32 to f32
  %nib_p = scalar.addf %nib_f, %c05 : f32
  %ds = scalar.mulf %d, %nib_p : f32
  %scale = scalar.mulf %ds, %c025 : f32
  %va = func.call @ggml_kquant_iq2_slot_values(%code_a, %sa) : (i32, i32) -> (vector<8xf32>)
  %vb = func.call @ggml_kquant_iq2_slot_values(%code_b, %sb) : (i32, i32) -> (vector<8xf32>)
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
out = fill('ggml_kquant_iq2xxs_grid_fill', grid('iq2xxs_grid', 256)) + fill('ggml_kquant_iq2xs_grid_fill', grid('iq2xs_grid', 512)) + common + dot_and_weights('iq2xxs') + dot_and_weights('iq2xs')
sys.stdout.write(out)
