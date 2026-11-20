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

"""Writes the content identity of a HIP kernel add-on tree, and leaves the file alone when it already says so.

usage: hip_addon_identity.py STAMP ADDON_DIR

The identity is the add-on directory's path followed by `<relative path> <sha256>` for every file the add-on
contributes to the build: kernels/* (the .hip sources and every header, .inc and manifest beside them), the
matchers' *.cpp and *.h, and addon.cmake. ggml-hrx-hip.cmake runs this at configure time and again in every
build; each code object and each matcher object depends on STAMP, so a tree whose content changed is rebuilt
whatever its mtimes say (a staged copy keeps the mtimes of the commit it was copied from), and a tree that did
not change costs one pass of hashing and rebuilds nothing, because STAMP keeps its mtime.
"""

import hashlib
import os
import sys


def identity(addon_dir: str) -> str:
    files = []
    kernels = os.path.join(addon_dir, "kernels")
    for name in sorted(os.listdir(kernels)):
        if os.path.isfile(os.path.join(kernels, name)):
            files.append("kernels/" + name)
    for name in sorted(os.listdir(addon_dir)):
        if name.endswith((".cpp", ".h")) or name == "addon.cmake":
            if os.path.isfile(os.path.join(addon_dir, name)):
                files.append(name)
    lines = [addon_dir]
    for rel in files:
        with open(os.path.join(addon_dir, rel), "rb") as handle:
            lines.append(f"{rel} {hashlib.sha256(handle.read()).hexdigest()}")
    return "\n".join(lines) + "\n"


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    stamp, addon_dir = sys.argv[1], sys.argv[2]
    try:
        text = identity(addon_dir)
    except OSError as error:
        print(f"hip_addon_identity: {error}", file=sys.stderr)
        return 1
    try:
        with open(stamp) as handle:
            if handle.read() == text:
                return 0
    except OSError:
        pass
    with open(stamp, "w") as handle:
        handle.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
