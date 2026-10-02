#include "dispatch-common.h"

#include "dispatch-binary.h"
#include "dispatch-copy.h"
#include "dispatch-flash-attention.h"
#include "dispatch-gated-mul-mat-id.h"
#include "dispatch-gated-mul-mat.h"
#include "dispatch-gather-add.h"
#include "dispatch-get-rows.h"
#include "dispatch-glu.h"
#include "dispatch-mul-mat-id.h"
#include "dispatch-mul-mat.h"
#include "dispatch-rmsnorm.h"
#include "dispatch-rope-set-rows.h"
#include "dispatch-scale.h"
#include "dispatch-grouped-mul-mat.h"
#include "dispatch-small-rows.h"
#include "dispatch-mul-mat-id-decode.h"
#include "dispatch-res-scale-pair.h"
#include "dispatch-hadamard.h"
#include "dispatch-zaya-cca-conv.h"
#include "dispatch-zaya-cca-qk-norm.h"
#include "dispatch-kquant-decode.h"
#include "dispatch-mul-mat-nvfp4.h"
#include "dispatch-add-id.h"
#include "dispatch-softplus.h"
#include "dispatch-swiglu-oai.h"
#include "dispatch-unary.h"

namespace ggml::hrx {

void register_common_dispatches(DispatchRegistryBuilder & registry) {
    register_binary_dispatch(registry);
    register_copy_dispatch(registry);
    register_flash_attention_dispatches(registry);
    register_gated_mul_mat_id_dispatches(registry);
    register_gated_mul_mat_dispatches(registry);
    register_gather_add_dispatch(registry);
    register_grouped_mul_mat_dispatch(registry);
    register_get_rows_dispatches(registry);
    register_glu_dispatches(registry);
    register_mul_mat_id_dispatches(registry);
    register_mul_mat_dispatches(registry);
    register_rope_set_rows_dispatches(registry);
    register_scale_dispatch(registry);
    register_small_rows_dispatches(registry);
    register_mul_mat_id_decode_dispatches(registry);
    register_res_scale_pair_dispatches(registry);
    register_hadamard_dispatches(registry);
    register_zaya_cca_conv_dispatches(registry);
    register_zaya_cca_qk_norm_dispatches(registry);
    register_kquant_decode_dispatches(registry);
    register_nvfp4_prefill_dispatches(registry);
    register_softplus_dispatches(registry);
    register_add_id_dispatches(registry);
    register_swiglu_oai_dispatches(registry);
    register_unary_dispatch(registry);
    register_rmsnorm_dispatches(registry);
}

}  // namespace ggml::hrx
