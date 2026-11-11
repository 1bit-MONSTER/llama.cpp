#pragma once

#include "graph.h"

#include <vector>

namespace ggml::hrx {

class GraphTraversalOrder {
  public:
    GraphTraversalOrder() = default;

    static GraphTraversalOrder build(const Graph & graph);

    const std::vector<const GraphNode *> & nodes() const { return nodes_; }

    // Nodes the dependency walk never reached, because an input of theirs never became available (a cycle, or a
    // node that waits for itself). They are appended in index order, which is a valid execution order but not the
    // one fused patterns are matched in: a pattern's root can then come before a node the pattern reads. Zero for
    // every graph llama.cpp builds today; engine #1925 was a non-zero count nobody could see.
    size_t stalled_count() const { return stalled_count_; }
    // index of the first such node; meaningful only when stalled_count() is not zero
    size_t first_stalled_node() const { return first_stalled_node_; }

  private:
    std::vector<const GraphNode *> nodes_;
    size_t                         stalled_count_      = 0;
    size_t                         first_stalled_node_ = 0;
};

}  // namespace ggml::hrx
