#include "graph.h"

#include "ggml-impl.h"

#include <cassert>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace ggml::hrx {
namespace {

// A value crosses the imported graph boundary (and therefore needs an external
// binding) in two cases: it is not produced by any node in the graph (a leaf such
// as a weight/token id, or an activation produced by another backend), or it is
// produced here but never consumed here (a graph output that must be downloaded).
// Only values produced and consumed inside the graph are true transients.
static bool tensor_is_external(const ggml_tensor *                             tensor,
                               const std::unordered_set<const ggml_tensor *> & node_outputs,
                               const std::unordered_map<const ggml_tensor *, int> & use_counts) {
    if (tensor->op == GGML_OP_NONE) {
        return true;
    }
    if (node_outputs.find(tensor) == node_outputs.end()) {
        return true;
    }
    const auto found = use_counts.find(tensor);
    return found == use_counts.end() || found->second == 0;
}

}  // namespace

GraphIndex GraphIndex::build(const Graph & graph) {
    GraphIndex                     index;
    const std::vector<GraphNode> & nodes = graph.nodes();
    for (size_t i = 0; i < nodes.size(); ++i) {
        const GraphNode & node = nodes[i];
        index.node_indices_.emplace(&node, i);
        index.producers_.emplace(node.output.value, &node);
        for (ValueId input : node.inputs) {
            index.consumers_[input.value].push_back(&node);
        }
    }
    return index;
}

const GraphNode * GraphIndex::producer(ValueId value) const {
    const auto found = producers_.find(value.value);
    return found == producers_.end() ? nullptr : found->second;
}

const std::vector<const GraphNode *> & GraphIndex::consumers(ValueId value) const {
    static const std::vector<const GraphNode *> empty;
    const auto                                  found = consumers_.find(value.value);
    return found == consumers_.end() ? empty : found->second;
}

bool GraphIndex::has_single_consumer(ValueId value) const {
    return consumers(value).size() == 1;
}

bool GraphIndex::node_index(const GraphNode * node, size_t & index) const {
    const auto found = node_indices_.find(node);
    if (found == node_indices_.end()) {
        return false;
    }
    index = found->second;
    return true;
}

Graph::Graph(const Graph & other) : values_(other.values_), nodes_(other.nodes_) {
    if (other.has_index()) {
        index_ = GraphIndex::build(*this);
    }
}

Graph & Graph::operator=(const Graph & other) {
    if (this == &other) {
        return *this;
    }
    values_ = other.values_;
    nodes_  = other.nodes_;
    index_.reset();
    if (other.has_index()) {
        index_ = GraphIndex::build(*this);
    }
    return *this;
}

Graph::Graph(Graph && other) : values_(std::move(other.values_)), nodes_(std::move(other.nodes_)) {
    if (other.has_index()) {
        index_ = GraphIndex::build(*this);
    }
}

Graph & Graph::operator=(Graph && other) {
    if (this == &other) {
        return *this;
    }
    values_ = std::move(other.values_);
    nodes_  = std::move(other.nodes_);
    index_.reset();
    if (other.has_index()) {
        index_ = GraphIndex::build(*this);
    }
    return *this;
}

GraphNode & Graph::add_node(ggml_op op, ValueId output, std::vector<ValueId> inputs) {
    index_.reset();
    GraphNode node;
    node.op     = op;
    node.output = output;
    node.inputs = std::move(inputs);
    nodes_.push_back(std::move(node));
    return nodes_.back();
}

Status Graph::build_index() {
    index_ = GraphIndex::build(*this);
    return {};
}

const GraphIndex & Graph::index() const {
    assert(index_.has_value());
    return *index_;
}

GraphImportResult import_ggml_graph(const ggml_cgraph & graph) {
    GraphImportResult                        result;
    std::unordered_set<const ggml_tensor *>  node_outputs;
    std::unordered_map<const ggml_tensor *, int> use_counts;
    for (int i = 0; i < graph.n_nodes; ++i) {
        const ggml_tensor * node = graph.nodes[i];
        if (node == nullptr) {
            result.status.log("ggml graph contains a null node");
            return result;
        }
        node_outputs.insert(node);
        for (const ggml_tensor * source : node->src) {
            if (source != nullptr) {
                ++use_counts[source];
            }
        }
    }

    ValueMap & values = result.graph.values();

    // Views share storage with the tensor they were created from, and the scheduler may
    // hand the HRX dispatcher a graph whose terminal value is such a view (e.g. a
    // standalone MUL_MAT followed by a RESHAPE that ggml_backend_sched kept on the same
    // backend). The classification must be consistent for the whole storage: if any
    // value in a storage is external (bound across the graph boundary), every value in
    // that storage is external too. Otherwise the kernel would write its result into a
    // transient arena slot while the consumer reads the host tensor the view aliases.
    auto storage_root_of = [](const ggml_tensor * tensor) {
        const ggml_tensor * root = tensor;
        while (root->view_src != nullptr) {
            root = root->view_src;
        }
        return root;
    };
    std::unordered_map<const ggml_tensor *, bool> storage_is_external;
    auto note_storage = [&](const ggml_tensor * tensor) {
        const bool          external = tensor_is_external(tensor, node_outputs, use_counts);
        const ggml_tensor * root     = storage_root_of(tensor);
        bool &              flag     = storage_is_external[root];
        flag                         = flag || external;
    };
    for (int i = 0; i < graph.n_nodes; ++i) {
        const ggml_tensor * node = graph.nodes[i];
        note_storage(node);
        for (const ggml_tensor * source : node->src) {
            if (source != nullptr) {
                note_storage(source);
            }
        }
    }
    auto value_kind = [&](const ggml_tensor * tensor) {
        const bool external =
            tensor_is_external(tensor, node_outputs, use_counts) || storage_is_external[storage_root_of(tensor)];
        return external ? ValueKind::External : ValueKind::Transient;
    };

    for (int i = 0; i < graph.n_nodes; ++i) {
        const ggml_tensor *  node = graph.nodes[i];
        std::vector<ValueId> inputs;
        for (const ggml_tensor * source : node->src) {
            if (source == nullptr) {
                continue;
            }
            inputs.push_back(values.get_or_add_tensor_value(source, value_kind(source)));
        }

        const ValueKind output_kind = value_kind(node);
        const ValueId   output      = values.get_or_add_tensor_value(node, output_kind);
        GraphNode &     graph_node  = result.graph.add_node(node->op, output, std::move(inputs));
        graph_node.params           = import_op_params(*node);
    }

    result.status.append(result.graph.build_index());
    return result;
}

bool is_layout_alias_op(ggml_op op) {
    return op == GGML_OP_VIEW || op == GGML_OP_RESHAPE || op == GGML_OP_PERMUTE || op == GGML_OP_TRANSPOSE;
}

bool is_layout_alias_node(const Graph & graph, const GraphNode & node) {
    return is_layout_alias_op(node.op) && node.inputs.size() == 1 &&
           graph.values().same_storage(node.output, node.inputs[0]);
}

}  // namespace ggml::hrx
