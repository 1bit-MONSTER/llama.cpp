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
"""check_stale_expectations.py -- keep `kRecordedStaleExpectations` honest, without running a kernel.

`tests/hrx-ops-test.cpp` records the kernel names that the suite demands and no matcher selects (engine#2774), so that
"the suite is 63 % red" reads as "these are known-stale, anything else is real". The table carries its own rot detector,
but it can only fire when a case RUNS and its name comes back -- a device run, minutes, and only for the cases a binary
reaches. The table's correctness is a property of the SOURCE: a recorded name must be absent from every
`GGML_HRX_KERNEL_REF` and present in a corpus manifest. This answers that statically, for every entry, in a second.

Exit codes: 0 every recorded name is unselectable and still exported, 1 an entry is wrong (see the message), 2 cannot
tell (a file this check needs is missing).

The two faults it reports are different mistakes:

  ROTTED      the name IS registered, so a matcher can select it -- the exception now suppresses a real mis-selection,
              which is the one thing the table must never do. (The runtime detector says "prune this line"; this says
              it at every pin, including the pins nobody runs the suite on.)
  NOT EXPORTED no corpus manifest defines the name any more, so the entry no longer describes "exported but
              unselected"; it describes a kernel that moved or was renamed, and the case it covers is unexplained.
"""
import argparse
import json
import re
import sys
from pathlib import Path

TABLE_RE = re.compile(r"kRecordedStaleExpectations\[\]\s*=\s*\{(.*?)\n\};", re.S)
ENTRY_RE = re.compile(r'"([a-z0-9_]+):([^"]+)"')
REF_RE = re.compile(r'GGML_HRX_KERNEL_REF\(\s*"([^"]+)"\s*,\s*"([^"]+)"\s*\)')


def read_records(root: Path):
    """(corpus, kernel) pairs from the harness table, or None when the table or the file is missing."""
    path = root / "tests" / "hrx-ops-test.cpp"
    if not path.is_file():
        return None, f"no {path}"
    match = TABLE_RE.search(path.read_text(encoding="utf-8", errors="replace"))
    if not match:
        return None, f"no kRecordedStaleExpectations table in {path}"
    return ENTRY_RE.findall(match.group(1)), None


def read_registrations(root: Path):
    """Every kernel name a dispatch registration can select, from any corpus. Any corpus matters: the table is written
    `loom_libs:<name>` throughout, but two of the names the harness deliberately excludes live under `qwen3_moe`, so a
    check that only reads the `loom_libs` prefix reports registrations as missing (I made that mistake by hand first)."""
    hx = root / "ggml" / "src" / "ggml-hrx"
    if not hx.is_dir():
        return None, f"no {hx}"
    names = set()
    for cpp in hx.rglob("*.cpp"):
        for _corpus, kernel in REF_RE.findall(cpp.read_text(encoding="utf-8", errors="replace")):
            names.add(kernel)
    return names, None


def read_exported(root: Path):
    """Every kernel name some corpus manifest defines."""
    corpus = root / "ggml" / "src" / "ggml-hrx" / "kernel-corpus" / "kernels"
    if not corpus.is_dir():
        return None, f"no {corpus}"
    names = set()

    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key in ("name", "symbol") and isinstance(value, str):
                    names.add(value)
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    for manifest in corpus.rglob("manifest.json"):
        try:
            walk(json.loads(manifest.read_text(encoding="utf-8", errors="replace")))
        except json.JSONDecodeError:
            return None, f"{manifest} is not JSON"
    return names, None


def check(root: Path):
    """-> (failures, cannot_tell). Each failure is (kind, corpus, kernel)."""
    records, why = read_records(root)
    if why:
        return [], why
    registered, why = read_registrations(root)
    if why:
        return [], why
    exported, why = read_exported(root)
    if why:
        return [], why
    failures = []
    for corpus, kernel in records:
        if kernel in registered:
            failures.append(("ROTTED", corpus, kernel))
        elif kernel not in exported:
            failures.append(("NOT EXPORTED", corpus, kernel))
    return failures, None


FIXTURE_TABLE = '''
static const char * const kRecordedStaleExpectations[] = {
    "loom_libs:ggml_mul_mat_tiled_input_f32_iq2_s_publish_f32",
    "loom_libs:ggml_some_kernel_that_vanished",
};
'''


def self_test(tmp: Path) -> int:
    """Hermetic: builds the three trees by hand, so no checkout is needed."""
    fails = 0

    def check_arm(name, ok, detail=""):
        nonlocal fails
        print(f"{'ok  ' if ok else 'FAIL'} {name}" + (f"  [{detail}]" if detail and not ok else ""))
        if not ok:
            fails += 1

    made = []

    def build(registered, exported, table=FIXTURE_TABLE):
        # A fresh tree per arm: a leftover dispatch-y.cpp from an earlier arm would answer for this one.
        root = tmp / f"repo{len(made)}"
        made.append(root)
        (root / "tests").mkdir(parents=True, exist_ok=True)
        (root / "tests" / "hrx-ops-test.cpp").write_text(table, encoding="utf-8")
        disp = root / "ggml" / "src" / "ggml-hrx" / "dispatch_registration"
        disp.mkdir(parents=True, exist_ok=True)
        body = "".join(f'GGML_HRX_KERNEL_REF("loom_libs", "{n}");\n' for n in registered)
        (disp / "dispatch-x.cpp").write_text(body, encoding="utf-8")
        corpus = root / "ggml" / "src" / "ggml-hrx" / "kernel-corpus" / "kernels" / "loom-libs"
        corpus.mkdir(parents=True, exist_ok=True)
        (corpus / "manifest.json").write_text(
            json.dumps({"kernels": [{"name": n, "symbol": n} for n in exported]}), encoding="utf-8")
        return root

    both = ["ggml_mul_mat_tiled_input_f32_iq2_s_publish_f32", "ggml_some_kernel_that_vanished"]

    # The good state: recorded, exported, unregistered.
    root = build([], both)
    got, why = check(root)
    check_arm("a recorded kernel that is exported and unregistered is clean", got == [] and why is None, str(why or got))

    # ROTTED: the dispatcher can now select it, so the exception hides a real mis-selection.
    root = build(["ggml_mul_mat_tiled_input_f32_iq2_s_publish_f32"], both)
    got, why = check(root)
    check_arm("a recorded kernel that IS registered is ROTTED",
              [k for k, _c, _n in got] == ["ROTTED"], str(got))

    # NOT EXPORTED: the entry no longer means "exported but unselected".
    root = build([], ["ggml_mul_mat_tiled_input_f32_iq2_s_publish_f32"])
    got, why = check(root)
    check_arm("a recorded kernel no manifest defines is NOT EXPORTED",
              [k for k, _c, _n in got] == ["NOT EXPORTED"], str(got))

    # The corpus-blindness that caught me: a name registered under another corpus still counts as registered.
    root = build([], both)
    (root / "ggml" / "src" / "ggml-hrx" / "dispatch_registration" / "dispatch-y.cpp").write_text(
        'GGML_HRX_KERNEL_REF("qwen3_moe", "ggml_mul_mat_tiled_input_f32_iq2_s_publish_f32");\n', encoding="utf-8")
    got, why = check(root)
    check_arm("a registration under another corpus still counts as registered",
              [k for k, _c, _n in got] == ["ROTTED"], str(got))

    # Cannot tell is its own answer and outranks a pass.
    root = build([], both)
    (root / "tests" / "hrx-ops-test.cpp").unlink()
    got, why = check(root)
    check_arm("a missing harness file cannot tell rather than passing", got == [] and why is not None, str(why))

    print("self-test ok" if not fails else f"self-test FAILED ({fails})")
    return 0 if not fails else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="check the tree at --root")
    ap.add_argument("--root", default=".", help="the llama.cpp checkout (default: .)")
    ap.add_argument("--self-test", action="store_true", help="hermetic fixtures, no checkout needed")
    args = ap.parse_args()
    if args.self_test:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            return self_test(Path(tmp))
    if not args.check:
        ap.print_usage()
        return 2
    failures, why = check(Path(args.root))
    if why:
        print(f"stale-expectations: cannot tell: {why}", file=sys.stderr)
        return 2
    for kind, corpus, kernel in failures:
        if kind == "ROTTED":
            print(f"  ROTTED {corpus}:{kernel}\n"
                  f"     a dispatch registration names it, so a matcher can select it: the exception is now hiding a "
                  f"real mis-selection (engine#2774). Prune the line, or the case should pass on its own.")
        else:
            print(f"  NOT EXPORTED {corpus}:{kernel}\n"
                  f"     no corpus manifest defines it, so 'the corpus still exports it and no matcher selects it' is "
                  f"no longer what this entry means: the kernel moved or was renamed (engine#2774).")
    if failures:
        return 1
    records, _ = read_records(Path(args.root))
    print(f"stale-expectations: {len(records)} recorded, every one exported by a corpus and selected by no matcher")
    return 0


if __name__ == "__main__":
    sys.exit(main())
