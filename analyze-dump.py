#!/usr/bin/env python3
# pi-hrx-sched, overnight #674: diff two llama-debug tensor dumps block by block and report the first value that
# diverges, plus a per-tensor magnitude ranking.
#
# A full-graph dump cannot be compared CPU-vs-device: the CPU arm runs f32 CPU kernels, the device arms run HRX kernels,
# so *every* tensor differs in the last bits. The comparison that isolates a placement corruption is between two device
# arms that differ only in placement (decline MUL_MAT = correct vs decline MUL_MAT,MUL = corrupt): same kernels, same
# weights, same schedule shape. Any large difference there is the bug.
#
# Usage: analyze-dump.py good.log bad.log [threshold]
import re
import sys

HDR = re.compile(r'common_debug_cb_eval:\s+(\S+)( \(view\))? = \((\w+)\)\s+(\w+)\((.*)\) = \{(.*)\}')
NUM = re.compile(r'-?\d+\.\d+(?:e[-+]?\d+)?|-?\d+(?:e[-+]?\d+)?')


def normalize(name):
    # the scheduler renames cross-placement values with CPU#<name>#<index>; that is placement bookkeeping, not content
    name = re.sub(r'CPU#([^#{}, ]+)#\d+', r'\1', name)
    name = re.sub(r'^CPU#', '', name)
    return name


def parse(path):
    blocks = []
    cur = None
    with open(path, encoding='utf-8', errors='replace') as f:
        for line in f:
            m = HDR.search(line)
            if m:
                cur = {'name': normalize(m.group(1)), 'type': m.group(3), 'op': m.group(4),
                       'srcs': normalize(m.group(5)), 'dims': m.group(6), 'vals': []}
                blocks.append(cur)
                continue
            if cur is not None:
                s = line.strip()
                if s.startswith('[') or s.startswith(']'):
                    cur['vals'].extend(float(x) for x in NUM.findall(s))
                elif '=' in line or line.startswith('ggml') or line.startswith('llama'):
                    cur = None
    return blocks


def main():
    good_p, bad_p = sys.argv[1], sys.argv[2]
    thr = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-3
    good, bad = parse(good_p), parse(bad_p)
    print(f'blocks: good={len(good)} bad={len(bad)}')
    n = min(len(good), len(bad))
    worst = []
    first = None
    for i in range(n):
        g, b = good[i], bad[i]
        if g['name'] != b['name'] or g['op'] != b['op']:
            print(f'block {i}: STRUCTURE differs: {g["name"]}/{g["op"]} vs {b["name"]}/{b["op"]} — alignment stops here')
            break
        if not g['vals'] or not b['vals']:
            continue
        k = min(len(g['vals']), len(b['vals']))
        dmax, rel, at = 0.0, 0.0, -1
        for j in range(k):
            d = abs(g['vals'][j] - b['vals'][j])
            scale = max(abs(g['vals'][j]), abs(b['vals'][j]), 1e-9)
            if d > dmax:
                dmax, rel, at = d, d / scale, j
        if dmax > 0:
            worst.append((rel, dmax, i, g['name'], g['op'], at, k))
        if first is None and rel > thr:
            first = (i, g['name'], g['op'], dmax, rel, at, k, g['dims'])
    for w in sorted(worst, reverse=True)[:12]:
        rel, dmax, i, name, op, at, k = w
        print(f'  rel={rel:9.4f} abs={dmax:12.5f} block {i:5d} {name:26s} {op}')
    if first:
        i, name, op, dmax, rel, at, k, dims = first
        print(f'\nFIRST DIVERGENCE above {thr:g}: block {i} {name} ({op}) dims={dims}')
        print(f'  element {at} of {k}: |Δ|={dmax:.6f} relative={rel:.6f}')
    else:
        print(f'\nno tensor differs by more than {thr:g} relative')


main()
