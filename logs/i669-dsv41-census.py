#!/usr/bin/env python3
# GGUF tensor census for the 217 GiB dsv41 shard set (engine #669 work item 4).
# Read-only, metadata only: GGUFReader mmaps, so this never loads tensor data.
# Shortcut: sizes are nbytes from tensor info; no quant dequant is attempted,
# good enough to decide which budgets decline which tensors.
import sys, glob, json
from gguf import GGUFReader

paths = sorted(glob.glob("/home/bcloud/models/dsv41/UD-IQ1_S/*.gguf"))
if len(sys.argv) > 1:
    paths = sys.argv[1:]
rows, total = [], 0
for p in paths:
    r = GGUFReader(p, mode="r")
    n = 0
    for t in r.tensors:
        nb = int(t.n_bytes)
        rows.append((t.name, nb))
        total += nb
        n += 1
    print(f"{p.split('/')[-1][-24:]}: {n} tensors")

rows.sort(key=lambda x: -x[1])
print(f"\ntensors={len(rows)}  total={total/2**30:.1f} GiB")

print("\nlargest 15:")
for name, nb in rows[:15]:
    print(f"  {nb/2**20:10.1f} MiB  {name}")

print("\ndeclines by budget (host_weight_exceeds_residency: nbytes > limit, whole MiB):")
for mb in (1, 2, 4, 8, 16, 32, 64, 128, 1024, 8192, 32768):
    sel = [(n, b) for n, b in rows if b > mb * 2**20]
    print(f"  {mb:6d} MiB -> {len(sel):4d} tensors declined, {sum(b for _, b in sel)/2**30:8.1f} GiB declines")
    if 0 < len(sel) <= 6:
        for n, b in sel:
            print(f"             {b/2**30:7.2f} GiB  {n}")
