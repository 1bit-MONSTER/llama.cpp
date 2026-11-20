// Copyright 2026 bong-water-water-bong
// SPDX-License-Identifier: Apache-2.0
//
// test-testing-suite-skip -- engine#2912: a device case that is skipped must not be indistinguishable from a pass.
//
// The caller this protects runs ONE case to test a hypothesis (`--hrx-test-case <name>`) and reads the exit code. On
// 2026-11-20, inside a held gpu-hold ticket, three device cases came back rc 0 in one second because another tenant was
// initialising into the GPU -- three green ticks over nothing, in the mode whose whole point is a quick honest answer.
// The runner now answers 77, the same "skipped" the engine's ctest rows use (SKIP_RETURN_CODE 77).
//
// No device is needed to run this test, which is the point: the fixture is "the suite cannot see a device", the same
// condition the bug appeared under, reachable on any host including CI.
#include "testing_suite.h"

#include <cstdio>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool ok, const std::string & what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) {
        failures++;
    }
}

}  // namespace

int main() {
    using namespace test_runner;

    Config config;
    config.suite_name = "skip-fixture";

    int ran_device = 0;  // a device case that runs must not run here at all
    int ran_host   = 0;
    std::vector<Case> cases;
    cases.push_back(Case{ "device-case", [&ran_device]() { ran_device++; }, /*requires_device=*/true, nullptr });
    cases.push_back(Case{ "host-case", [&ran_host]() { ran_host++; }, /*requires_device=*/false, nullptr });

    // THE ARM: a device case with no device is a skip, and a skip is not a pass.
    const int skipped = run_single_case_in_process(config, cases, "device-case", /*has_device=*/false);
    check(skipped == 77, "a device case with no device returns 77, not 0 (engine#2912)");
    check(ran_device == 0, "and it does not run the case body");

    // 77 must mean "skipped", not "anything that did not pass": a host case with no device still passes, and an
    // unknown name is still a usage error (2). Without these two, returning 77 unconditionally would look correct.
    check(run_single_case_in_process(config, cases, "host-case", /*has_device=*/false) == 0,
          "a host case still returns 0 without a device");
    check(ran_host == 1, "and a host case does run its body");
    check(run_single_case_in_process(config, cases, "no-such-case", /*has_device=*/false) == 2,
          "an unknown case name is still 2, not 77");

    // And a device case WITH a device is a real run, so the 77 is conditioned on the device and not on the flag alone.
    check(run_single_case_in_process(config, cases, "device-case", /*has_device=*/true) == 0,
          "a device case with a device returns 0");
    check(ran_device == 1, "and that one does run its body");

    // engine#2393's guard, pinned from this side: seven HRX ctest rows set
    // FAIL_REGULAR_EXPRESSION "test skipped: no device available" so that a device-less run of a hardware suite FAILS
    // rather than passing. That substring has to survive every edit to the skip line -- the first version of this
    // change put "(77)" inside it and turned the guard into dead text that still read like a guard.
    check(std::string(kSkipMessagePrefix) == "test skipped: no device available",
          "the skip message keeps the substring seven HRX rows match in FAIL_REGULAR_EXPRESSION (engine#2393)");

    std::printf("%s\n", failures ? "FAILED" : "all cases ok");
    return failures ? 1 : 0;
}
