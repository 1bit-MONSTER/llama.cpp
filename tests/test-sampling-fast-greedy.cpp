// common_sampler greedy fast path (llama.cpp#385): the eligibility predicate turns the fast path off for every stage that
// can move the argmax, and on the same logits the fast-path argmax picks the token the equivalent sampler chain picks.
// Offline: no model, the chain is built from llama.h samplers.

#include "sampling.h"
#include "llama.h"

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

int main() {
    test_eligibility();
    test_equivalence();
    test_nan_and_empty();
    if (failures != 0) {
        std::fprintf(stderr, "test-sampling-fast-greedy: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("test-sampling-fast-greedy: OK\n");
    return 0;
}
