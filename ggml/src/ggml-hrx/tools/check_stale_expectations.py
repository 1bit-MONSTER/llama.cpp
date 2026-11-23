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
"""check_stale_expectations.py -- every kernel name the HRX op suites expect must be one dispatch can select.

engine#2774: `tests/hrx-ops-test.cpp` asserts which kernels a graph schedules by name (`"loom_libs:<kernel>"`). A name
that no `GGML_HRX_KERNEL_REF` registers can never be scheduled, so a positive expectation on it aborts every run and a
negative one (`== end()`) is true by construction. engine#3408 added `tests/test-hrx-ops.cpp`, the second suite, which
had two such names. The suite used to carry a `kRecordedStaleExpectations` table that
silently skipped such names; at 7737fa364 it hid 186 checks in 170 cases. That table is gone, and this check keeps the
state it left: it answers from the source, without a device, in about a second.

  UNSELECTABLE  a `<corpus>:<kernel>` literal in the suite that no registration names. Rewrite the expectation to the
                kernel the case schedules (the device run prints the sequence on a miss), or delete the check.
  TABLE         the suite defines `kRecordedStaleExpectations` again. Fix the expectation instead.

Comments are not read, and `hip:` names are content-addressed add-on objects, not registrations, so they are not
checked here.

Exit codes: 0 every literal is selectable, 1 a fault (see the messages), 2 cannot tell (a file is missing, or the suite
yields no literal at all, which means this parser no longer matches it).
"""
import argparse
import json
import re
import sys
from pathlib import Path

SUITES = ("tests/hrx-ops-test.cpp", "tests/test-hrx-ops.cpp")  # engine#3408: both HRX op suites
CORPORA = ("loom_libs", "qwen3_moe", "hrx")
LITERAL_RE = re.compile(r'"((?:' + "|".join(CORPORA) + r'):[A-Za-z0-9_]+)"')
REF_RE = re.compile(r'GGML_HRX_KERNEL_REF\(\s*"([^"]+)"\s*,\s*"([^"]+)"\s*\)')
TABLE_RE = re.compile(r"\bkRecordedStaleExpectations\s*\[\s*\]\s*=")


def strip_comments(text: str) -> str:
    """Blank out // and /* */ comments, keeping string literals and line numbers."""
    out = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c == '"':
            j = i + 1
            while j < n and text[j] != '"':
                j += 2 if text[j] == "\\" else 1
            out.append(text[i:j + 1])
            i = j + 1
        elif text.startswith("//", i):
            j = text.find("\n", i)
            i = n if j < 0 else j
        elif text.startswith("/*", i):
            j = text.find("*/", i + 2)
            j = n if j < 0 else j + 2
            out.append("\n" * text.count("\n", i, j))
            i = j
        else:
            out.append(c)
            i += 1
    return "".join(out)


def read_registrations(root: Path):
    """Every `<corpus>:<kernel>` a dispatch registration can select. Headers count: the q8 quantize pass is
    registered from dispatch-mul-mat-common.h, and a .cpp-only scan reports it as missing."""
    hx = root / "ggml" / "src" / "ggml-hrx"
    if not hx.is_dir():
        return None, f"no {hx}"
    names = set()
    for pattern in ("*.cpp", "*.h"):
        for path in hx.rglob(pattern):
            if "kernel-corpus" in path.parts:
                continue
            for corpus, kernel in REF_RE.findall(path.read_text(encoding="utf-8", errors="replace")):
                names.add(f"{corpus}:{kernel}")
    return names, None


def check(root: Path, suites=SUITES):
    """-> (faults, literal_count, cannot_tell). A fault is (kind, file, line, name)."""
    registered, why = read_registrations(root)
    if why:
        return [], 0, why
    faults = []
    count = 0
    for rel in suites:
        path = root / rel
        if not path.is_file():
            return [], 0, f"no {path}"
        code = strip_comments(path.read_text(encoding="utf-8", errors="replace"))
        for match in TABLE_RE.finditer(code):
            faults.append(("TABLE", rel, code.count("\n", 0, match.start()) + 1, "kRecordedStaleExpectations"))
        for match in LITERAL_RE.finditer(code):
            count += 1
            if match.group(1) not in registered:
                faults.append(("UNSELECTABLE", rel, code.count("\n", 0, match.start()) + 1, match.group(1)))
    if count == 0:
        return [], 0, "no <corpus>:<kernel> literal found in the suite: the parser no longer matches it"
    return faults, count, None


def self_test(tmp: Path) -> int:
    """Hermetic: builds each tree by hand, so no checkout is needed."""
    fails = 0

    def arm(name, ok, detail=""):
        nonlocal fails
        print(f"{'ok  ' if ok else 'FAIL'} {name}" + (f"  [{detail}]" if detail and not ok else ""))
        if not ok:
            fails += 1

    made = []
    one = ("tests/hrx-ops-test.cpp",)

    def check1(root):
        return check(root, one)

    def build(suite, refs, header_refs=()):
        root = tmp / f"repo{len(made)}"
        made.append(root)
        (root / "tests").mkdir(parents=True, exist_ok=True)
        (root / "tests" / "hrx-ops-test.cpp").write_text(suite, encoding="utf-8")
        disp = root / "ggml" / "src" / "ggml-hrx" / "dispatch_registration"
        disp.mkdir(parents=True, exist_ok=True)
        body = "".join(f'GGML_HRX_KERNEL_REF("{c}", "{k}");\n' for c, k in refs)
        (disp / "dispatch-x.cpp").write_text(body, encoding="utf-8")
        hbody = "".join(f'constexpr auto k = GGML_HRX_KERNEL_REF("{c}", "{k}");\n' for c, k in header_refs)
        (disp / "dispatch-x.h").write_text(hbody, encoding="utf-8")
        return root

    good = 'require({ "loom_libs:ggml_binary_f32", "hip:k_0123" });\n'
    refs = [("loom_libs", "ggml_binary_f32")]

    faults, count, why = check1(build(good, refs))
    arm("a registered literal is clean, and a hip: name is not checked", faults == [] and count == 1 and why is None,
        str((faults, count, why)))

    faults, _, why = check1(build(good + 'require({ "loom_libs:ggml_gone_f32" });\n', refs))
    arm("an unregistered literal is UNSELECTABLE, on its line",
        [(k, l, n) for k, _f, l, n in faults] == [("UNSELECTABLE", 2, "loom_libs:ggml_gone_f32")], str(faults))

    faults, _, why = check1(build(good + '// was "loom_libs:ggml_gone_f32"\n/* "loom_libs:ggml_gone_f32" */\n', refs))
    arm("a literal inside a comment is not read", faults == [] and why is None, str((faults, why)))

    faults, _, why = check1(build('x({ "qwen3_moe:ggml_quantize_q8_1_x4_f32" });\n', [],
                                 [("qwen3_moe", "ggml_quantize_q8_1_x4_f32")]))
    arm("a registration in a header counts", faults == [] and why is None, str((faults, why)))

    faults, _, why = check1(build('x({ "qwen3_moe:ggml_binary_f32" });\n', refs))
    arm("the corpus is part of the name: loom_libs:x does not register qwen3_moe:x",
        [k for k, *_ in faults] == ["UNSELECTABLE"], str(faults))

    faults, _, why = check1(build(good + 'static const char * const kRecordedStaleExpectations[] = { };\n', refs))
    arm("a suppression table coming back is TABLE", [k for k, *_ in faults] == ["TABLE"], str(faults))

    faults, _, why = check1(build("int main() { return 0; }\n", refs))
    arm("a suite with no literal at all cannot tell rather than passing", faults == [] and why is not None, str(why))

    root = build(good, refs)
    (root / "tests" / "hrx-ops-test.cpp").unlink()
    faults, _, why = check1(root)
    arm("a missing suite cannot tell rather than passing", faults == [] and why is not None, str(why))

    root = build(good, refs)
    (root / "tests" / "test-hrx-ops.cpp").write_text('x({ "loom_libs:ggml_gone_f32" });\n', encoding="utf-8")
    faults, _, why = check(root)
    arm("the second suite (tests/test-hrx-ops.cpp) is read too (engine#3408)",
        [(k, f, n) for k, f, _l, n in faults]
        == [("UNSELECTABLE", "tests/test-hrx-ops.cpp", "loom_libs:ggml_gone_f32")],
        str((faults, why)))

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
    faults, count, why = check(Path(args.root))
    if why:
        print(f"stale-expectations: cannot tell: {why}", file=sys.stderr)
        return 2
    for kind, rel, line, name in faults:
        if kind == "TABLE":
            print(f"  TABLE {rel}:{line}\n     the suite defines kRecordedStaleExpectations again: rewrite the "
                  f"expectation to the kernel the case schedules instead of suppressing it (engine#2774).")
        else:
            print(f"  UNSELECTABLE {rel}:{line} {name}\n     no GGML_HRX_KERNEL_REF registers it, so no graph can "
                  f"schedule it: name the kernel the case schedules, or drop the check (engine#2774).")
    if faults:
        return 1
    print(f"stale-expectations: {count} expectation literal(s), every one selectable by a registration")
    return 0


if __name__ == "__main__":
    sys.exit(main())
