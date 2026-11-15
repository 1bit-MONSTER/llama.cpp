// common_sampler greedy fast path (llama.cpp#385): the eligibility predicate turns the fast path off for every stage that
// can move the argmax, and on the same logits the fast-path argmax picks the token the equivalent sampler chain picks.
// Offline: no model, the chain is built from llama.h samplers.

#include "sampling.h"
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

static int failures = 0;

#define CHECK(cond)                                                                    \
    do {                                                                               \
        if (!(cond)) {                                                                 \
            std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); \
            ++failures;                                                                \
        }                                                                              \
    } while (0)

static common_params_sampling greedy_params() {
    common_params_sampling p;
    p.temp = 0.0f;  // the default sampler order ends in TEMPERATURE
    return p;
}

static void test_eligibility() {
    {
        const common_params_sampling p = greedy_params();
        CHECK(common_sampler_fast_greedy_eligible(p, false, false));
        CHECK(!common_sampler_fast_greedy_eligible(p, true, false));   // grammar
        CHECK(!common_sampler_fast_greedy_eligible(p, false, true));   // reasoning budget
    }
    auto off = [](void (*edit)(common_params_sampling &)) {
        common_params_sampling p = greedy_params();
        edit(p);
        return !common_sampler_fast_greedy_eligible(p, false, false);
    };
    CHECK(off([](common_params_sampling & p) { p.temp = 0.7f; }));
    CHECK(off([](common_params_sampling & p) { p.dynatemp_range = 0.5f; }));
    CHECK(off([](common_params_sampling & p) { p.mirostat = 2; }));
    CHECK(off([](common_params_sampling & p) { p.n_probs = 1; }));
    CHECK(off([](common_params_sampling & p) { p.backend_sampling = true; }));
    CHECK(off([](common_params_sampling & p) { p.xtc_probability = 0.5f; }));
    CHECK(off([](common_params_sampling & p) { p.typ_p = 0.9f; }));
    CHECK(off([](common_params_sampling & p) { p.min_p = 1.5f; }));
    CHECK(off([](common_params_sampling & p) { p.dry_multiplier = 0.8f; }));
    CHECK(off([](common_params_sampling & p) { p.penalty_repeat = 1.1f; }));
    CHECK(off([](common_params_sampling & p) { p.logit_bias.push_back({ 7, 2.5f }); }));       // finite bias
    CHECK(off([](common_params_sampling & p) { p.samplers.push_back(COMMON_SAMPLER_TYPE_TOP_K); }));  // temp not last
    {
        common_params_sampling p = greedy_params();
        p.logit_bias.push_back({ 7, -INFINITY });  // a suppression is a mask, not a reason to fall back
        CHECK(common_sampler_fast_greedy_eligible(p, false, false));
        p.penalty_last_n = 0;
        p.penalty_repeat = 1.3f;  // penalties over zero tokens do nothing
        CHECK(common_sampler_fast_greedy_eligible(p, false, false));
    }
}

// the chain common_sampler_init builds for greedy_params(), minus the stages that are identity at their defaults
static llama_token chain_pick(const std::vector<float> & logits, const std::vector<llama_logit_bias> & bias) {
    const int32_t n = (int32_t) logits.size();
    llama_sampler * chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
    if (!bias.empty()) {
        llama_sampler_chain_add(chain, llama_sampler_init_logit_bias(n, (int32_t) bias.size(), bias.data()));
    }
    llama_sampler_chain_add(chain, llama_sampler_init_top_k(40));
    llama_sampler_chain_add(chain, llama_sampler_init_top_p(0.95f, 1));
    llama_sampler_chain_add(chain, llama_sampler_init_min_p(0.05f, 1));
    llama_sampler_chain_add(chain, llama_sampler_init_temp_ext(0.0f, 0.0f, 1.0f));
    llama_sampler_chain_add(chain, llama_sampler_init_greedy());

    std::vector<llama_token_data> cur(n);
    for (int32_t i = 0; i < n; ++i) {
        cur[i] = { i, logits[i], 0.0f };
    }
    llama_token_data_array cur_p = { cur.data(), cur.size(), -1, false };
    llama_sampler_apply(chain, &cur_p);
    const llama_token id = cur_p.data[cur_p.selected].id;
    llama_sampler_free(chain);
    return id;
}

static void test_equivalence() {
    std::mt19937                          rng(1234);
    std::uniform_real_distribution<float> dist(-8.0f, 8.0f);
    const int32_t                         n = 4096;
    for (int trial = 0; trial < 200; ++trial) {
        std::vector<float> logits(n);
        for (auto & v : logits) {
            v = dist(rng);
        }
        // plain greedy
        CHECK(common_sampler_fast_greedy_argmax(logits.data(), n, nullptr) == chain_pick(logits, {}));

        // a -inf bias on the argmax: both paths must move to the runner-up
        const int32_t     top = common_sampler_fast_greedy_argmax(logits.data(), n, nullptr);
        std::vector<char> mask(n, 0);
        mask[top] = 1;
        const int32_t fast = common_sampler_fast_greedy_argmax(logits.data(), n, mask.data());
        CHECK(fast != top);
        CHECK(fast == chain_pick(logits, { { top, -INFINITY } }));
    }
}

static void test_nan_and_empty() {
    std::vector<float> logits = { 0.5f, 2.0f, NAN, 1.0f };
    CHECK(common_sampler_fast_greedy_argmax(logits.data(), (int32_t) logits.size(), nullptr) == -1);  // full path reports it

    std::vector<float> all_masked = { 1.0f, 2.0f };
    std::vector<char>  mask       = { 1, 1 };
    CHECK(common_sampler_fast_greedy_argmax(all_masked.data(), 2, mask.data()) == -1);  // nothing left: full path decides
}

// the scalar contract the vectorised path must reproduce exactly (lowest index on ties, masks skipped, NaN -> -1)
static int32_t reference_argmax(const float * logits, int32_t n, const char * mask) {
    int32_t best   = -1;
    float   best_v = -INFINITY;
    for (int32_t i = 0; i < n; ++i) {
        if (std::isnan(logits[i])) {
            return -1;
        }
        if (logits[i] > best_v && (mask == nullptr || !mask[i])) {
            best_v = logits[i];
            best   = i;
        }
    }
    return best;
}

// common_sampler_fast_greedy_argmax dispatches to an AVX2 body on x86 hosts that have it (C4-STREAMS); it must agree
// with the scalar contract on every input shape: ties, masks over the maximum, NaN / inf, and lengths that are not a
// multiple of the vector width.
static void test_vector_matches_scalar() {
    std::mt19937                          rng(4321);
    std::uniform_real_distribution<float> dist(-20.0f, 20.0f);
    std::vector<float>                    logits;
    std::vector<char>                     mask;
    for (int trial = 0; trial < 120; ++trial) {
        const int32_t n = 248320 + trial % 8;  // every remainder mod the vector width: the tail runs on 7 of 8
        logits.assign(n, 0.0f);
        mask.assign(n, 0);
        for (auto & v : logits) {
            v = dist(rng);
        }
        std::fill(mask.begin(), mask.end(), 0);
        if (trial % 3 == 1) {
            for (int k = 0; k < 50; ++k) {
                mask[rng() % n] = 1;
            }
        }
        if (trial % 5 == 2) {  // exact tie: the lowest id wins
            logits[rng() % n] = 50.0f;
            logits[rng() % n] = 50.0f;
        }
        if (trial % 7 == 3) {  // the global maximum is masked: the next one wins
            const int32_t a = rng() % n;
            logits[a]       = 60.0f;
            mask[a]         = 1;
        }
        if (trial % 17 == 6) {  // a tie where one copy is masked
            const int32_t a = rng() % n;
            logits[a]       = 70.0f;
            mask[a]         = 1;
            logits[rng() % n] = 70.0f;
        }
        if (trial % 11 == 4) {
            logits[rng() % n] = NAN;
        }
        if (trial % 13 == 5) {
            logits[rng() % n] = INFINITY;
        }
        if (trial % 6 == 0) {  // ties across lane / vector boundaries and on the last element (in the tail)
            const int32_t b = 8 * (int32_t) (rng() % (uint32_t) (n / 8 - 2)) + 7;
            logits[b] = logits[b + 1] = logits[b + 9] = logits[n - 1] = 80.0f;
        }
        if (trial % 9 == 7) {  // the last element alone holds the maximum
            logits[n - 1] = 90.0f;
        }
        if (trial % 10 == 8) {  // a vocabulary ban: a large adjacent block at -inf, the max just after it
            const int32_t a = (int32_t) (rng() % (uint32_t) (n - 70000));
            std::fill(logits.begin() + a, logits.begin() + a + 65536, -INFINITY);
            logits[a + 65536] = 85.0f;
        }
        const char *  m   = (trial % 2) ? mask.data() : nullptr;
        const int32_t ref = reference_argmax(logits.data(), n, m);
        CHECK(common_sampler_fast_greedy_argmax(logits.data(), n, m) == ref);
        CHECK(common_sampler_fast_greedy_argmax_scalar(logits.data(), n, m) == ref);  // the non-AVX2 path, run here too
    }
    for (const int32_t k : { 1, 7, 8, 9, 15, 16, 17, 31 }) {  // the vector tail
        std::vector<float> s(k);
        for (auto & v : s) {
            v = dist(rng);
        }
        CHECK(common_sampler_fast_greedy_argmax(s.data(), k, nullptr) == reference_argmax(s.data(), k, nullptr));
        CHECK(common_sampler_fast_greedy_argmax_scalar(s.data(), k, nullptr) == reference_argmax(s.data(), k, nullptr));
    }
    std::vector<float> all_inf(16, -INFINITY);
    CHECK(common_sampler_fast_greedy_argmax(all_inf.data(), 16, nullptr) == -1);
    CHECK(common_sampler_fast_greedy_argmax_scalar(all_inf.data(), 16, nullptr) == -1);
}

int main() {
    test_eligibility();
    test_vector_matches_scalar();
    test_equivalence();
    test_nan_and_empty();
    if (failures != 0) {
        std::fprintf(stderr, "test-sampling-fast-greedy: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("test-sampling-fast-greedy: OK\n");
    return 0;
}
