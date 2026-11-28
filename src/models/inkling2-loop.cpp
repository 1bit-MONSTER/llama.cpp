#include "models.h"

int llama_model_inkling2_loop::layer_phys(int il) const {
    return il < n_phys ? il : core_start + (il - n_phys) % (core_end - core_start);
}

void llama_model_inkling2_loop::load_arch_hparams(llama_model_loader & ml) {
    llama_model_inkling2::load_arch_hparams(ml);

    // two spellings of the same metadata: our converter writes loop.start/loop.end/loop.count,
    // the upstream inkling2 export writes loop.core_start/loop.core_end/num_loops
    if (!ml.get_key(LLM_KV_LOOP_START,     core_start, false)) { ml.get_key("inkling2.loop.core_start", core_start); }
    if (!ml.get_key(LLM_KV_LOOP_END,       core_end,   false)) { ml.get_key("inkling2.loop.core_end",   core_end);   }
    if (!ml.get_key(LLM_KV_LOOP_COUNT,     n_passes,   false)) { ml.get_key(LLM_KV_NUM_LOOPS,          n_passes);   }
    ml.get_key(LLM_KV_LOOP_GATE_RANK,     gate_rank);
    ml.get_key(LLM_KV_LOOP_VARYING,       varying, false);

    n_phys = hparams.n_layer();

    // block_count may count executed blocks (physical + the repeated core) instead of weight sets:
    // when removing the repetitions lands on a real physical count, that is the convention in use
    if (n_passes > 1 && core_end > core_start && n_phys > (n_passes - 1)*(core_end - core_start)) {
        const uint32_t phys = n_phys - (n_passes - 1)*(core_end - core_start);
        if (phys > core_end && phys < n_phys) {
            LLAMA_LOG_INFO("%s: block_count %u counts executed blocks, %u weight sets\n", __func__, n_phys, phys);
            n_phys = phys;
            hparams.n_layer_all = phys;
        }
    }
    const uint32_t block = hparams.attn_res_block_size;
    // a fixed loop needs at least 2 passes, a variable loop at least 1
    if (n_passes < (varying ? 1u : 2u) || !gate_rank || !core_start || core_start >= core_end || core_end > (uint32_t) n_phys ||
        core_start % block || core_end % block || hparams.n_layer_nextn) {
        throw std::runtime_error(format("inkling2_loop: invalid loop core [%u, %u) with block size %u, %u passes or gate rank %u for %d layers",
                core_start, core_end, block, n_passes, gate_rank, n_phys));
    }
    const size_t n_slot = n_phys + (size_t) (n_passes - 1)*(core_end - core_start);
    if (n_slot > LLAMA_MAX_LAYERS) {
        throw std::runtime_error(format("inkling2_loop: %zu cache slots exceed the maximum of %d", n_slot, LLAMA_MAX_LAYERS));
    }

    // expand to the cache slots before load_tensors() allocates the layers and the memory
    hparams.n_layer_all = n_slot;
    for (int il = n_phys; il < (int) n_slot; ++il) {
        const int p = layer_phys(il);
        hparams.n_head_arr[il]        = hparams.n_head_arr[p];
        hparams.n_head_kv_arr[il]     = hparams.n_head_kv_arr[p];
        hparams.n_ff_arr[il]          = hparams.n_ff_arr[p];
        hparams.n_ff_exp_arr[il]      = hparams.n_ff_exp_arr[p];
        hparams.n_expert_used_arr[il] = hparams.n_expert_used_arr[p];
        hparams.is_swa_impl[il]       = hparams.is_swa_impl[p];
        hparams.is_recr_impl[il]      = hparams.is_recr_impl[p];
    }
}

void llama_model_inkling2_loop::load_arch_tensors(llama_model_loader & ml) {
    llama_model_inkling2::load_arch_tensors(ml);

    const int64_t n_embd = hparams.n_embd;
    const int64_t rank   = gate_rank;
    const llm_tensor down[LOOP_CTRL_COUNT]   = { LLM_TENSOR_LOOP_Q_CTRL_DOWN,   LLM_TENSOR_LOOP_K_CTRL_DOWN,   LLM_TENSOR_LOOP_V_CTRL_DOWN,   LLM_TENSOR_LOOP_FFN_CTRL_DOWN   };
    const llm_tensor gain[LOOP_CTRL_COUNT]   = { LLM_TENSOR_LOOP_Q_CTRL_GAIN,   LLM_TENSOR_LOOP_K_CTRL_GAIN,   LLM_TENSOR_LOOP_V_CTRL_GAIN,   LLM_TENSOR_LOOP_FFN_CTRL_GAIN   };
    const llm_tensor forget[LOOP_CTRL_COUNT] = { LLM_TENSOR_LOOP_Q_CTRL_FORGET, LLM_TENSOR_LOOP_K_CTRL_FORGET, LLM_TENSOR_LOOP_V_CTRL_FORGET, LLM_TENSOR_LOOP_FFN_CTRL_FORGET };

    // the upstream inkling2 export spells the same four controllers loop_attn_/loop_k_/loop_v_/loop_mlp_*
    const char * const alt[LOOP_CTRL_COUNT] = { "attn", "k", "v", "mlp" };

    // ... and keeps the q and kv queries in two global [n_embd, 2*n_core] matrices, two rows per core layer
    ggml_tensor * msg_res = nullptr;
    ggml_tensor * msg_kv  = nullptr;

    loop_layers.resize(core_end - core_start);
    for (uint32_t il = core_start; il < core_end; ++il) {
        auto & l = loop_layers[il - core_start];
        // the q, k, v and ffn message queries
        l.msg_query = create_tensor(tn(LLM_TENSOR_LOOP_MSG_QUERY, "weight", il), {n_embd, LOOP_CTRL_COUNT}, TENSOR_NOT_REQUIRED);
        if (!l.msg_query) {
            if (!msg_res) {
                const int64_t n_core = core_end - core_start;
                msg_res = create_tensor(tn(LLM_TENSOR_LOOP_RES_QUERY,    "weight"), {n_embd, 2*n_core}, 0);
                msg_kv  = create_tensor(tn(LLM_TENSOR_LOOP_KV_RES_QUERY, "weight"), {n_embd, 2*n_core}, 0);
            }
            l.msg_res = msg_res;
            l.msg_kv  = msg_kv;
        }
        auto load_ctrl = [&](int c, llm_tensor t, const char * kind, std::initializer_list<int64_t> ne) {
            ggml_tensor * w = create_tensor(tn(t, "weight", il), ne, TENSOR_NOT_REQUIRED);
            if (w) {
                return w;
            }
            const std::string name = format("blk.%d.loop_%s_%s.weight", il, alt[c], kind);
            return create_tensor(LLM_TN_IMPL(name.c_str()), ne, 0);
        };
        for (int c = 0; c < LOOP_CTRL_COUNT; ++c) {
            l.ctrl_down[c]   = load_ctrl(c, down[c],   "down", {2*n_embd, rank});
            l.ctrl_gain[c]   = load_ctrl(c, gain[c],   "gain", {rank, n_embd});
            l.ctrl_forget[c] = load_ctrl(c, forget[c], "forget", {rank, n_embd});
        }
    }
}

void llama_model_inkling2_loop::post_arch_tensors(llama_model_loader &) {
    // share the physical weights, each cache slot keeps its own KV and conv state
    for (int il = n_phys; il < (int) hparams.n_layer(); ++il) {
        layers[il] = layers[layer_phys(il)];
    }
}

std::unique_ptr<llm_graph_context> llama_model_inkling2_loop::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

// the four queries of loop layer i: the file's own loop_msg_query when it has one, else the upstream export's
// two global matrices, whose two rows per layer interleave as [q[2i], kv[2i], kv[2i+1], q[2i+1]]
ggml_tensor * llama_model_inkling2_loop::graph::loop_msg(const loop_layer & loop, int i) {
    if (loop.msg_query) {
        return loop.msg_query;
    }

    auto row = [&](ggml_tensor * t, int64_t r) {
        return ggml_view_2d(ctx0, t, n_embd, 1, t->nb[1], r*t->nb[1]);
    };
    ggml_tensor * qk = ggml_concat(ctx0, row(loop.msg_res, 2*i),     row(loop.msg_kv, 2*i),     1);
    ggml_tensor * vf = ggml_concat(ctx0, row(loop.msg_kv, 2*i + 1), row(loop.msg_res, 2*i + 1), 1);

    return ggml_concat(ctx0, qk, vf, 1);
}

llama_model_inkling2_loop::graph::graph(const llama_model_inkling2_loop & m, const llm_graph_params & params) : llama_model_inkling2::graph(m, params, no_build{}) {
    const int start = m.core_start;
    const int end   = m.core_end;

    auto * cur = build_inputs();
    std::vector<ggml_tensor *> blocks;

    // DFlash2/DSpark target features are indexed by layer visit, not by cache slot
    int visit = 0;

    for (int il = 0; il < start; ++il) {
        res->t_layer_inp[visit++] = cur;
        cur = build_layer(il, il, cur, blocks);
    }

    // every pass starts from the prelude output
    auto * prelude = cur;
    const auto prelude_blocks = blocks;
    for (uint32_t pass = 0; pass < m.n_passes; ++pass) {
        // the bank of the previous pass: its blocks after the first core block, and its output
        std::vector<ggml_tensor *> bank;
        ggml_tensor * bank_out = nullptr;
        if (pass > 0) {
            bank.assign(blocks.begin() + prelude_blocks.size() + 1, blocks.end());
            bank_out = cur;
            cur = prelude;
            blocks = prelude_blocks;
        }
        for (int ip = start; ip < end; ++ip) {
            const int il = pass == 0 ? ip : m.n_phys + (pass - 1)*(end - start) + (ip - start);
            res->t_layer_inp[visit++] = cur;
            if (pass == 0) {
                cur = build_layer(il, ip, cur, blocks);
                continue;
            }
            const auto & loop = m.loop_layers[ip - start];
            cur = build_layer(il, ip, cur, blocks, mix(bank, bank_out, loop_msg(loop, ip - start)), &loop);
        }
    }

    for (int il = end; il < m.n_phys; ++il) {
        res->t_layer_inp[visit++] = cur;
        cur = build_layer(il, il, cur, blocks);
    }
    GGML_ASSERT(visit == n_layer);
    build_output(cur, blocks);
}
