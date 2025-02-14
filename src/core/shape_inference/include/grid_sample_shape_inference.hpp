// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include "openvino/op/grid_sample.hpp"
#include "utils.hpp"

namespace ov {
namespace op {
namespace v9 {

template <class TShape, class TRShape = result_shape_t<TShape>>
std::vector<TRShape> shape_infer(const GridSample* op, const std::vector<TShape>& input_shapes) {
    NODE_VALIDATION_CHECK(op,
                          input_shapes.size() == 2,
                          "Incorrect number of input shapes in GridSample's shape inference function");
    const auto& data_shape = input_shapes[0];
    const auto& grid_shape = input_shapes[1];
    const auto data_rank = data_shape.rank().get_length();
    const auto grid_rank = grid_shape.rank().get_length();
    const auto data_dims = data_rank - 2;
    const auto last_dim = grid_rank - 1;

    NODE_VALIDATION_CHECK(op,
        data_rank == grid_rank && (data_rank == 5 || data_rank == 4),
        "The dimension of data and grid tensor's shape should be the same, and must be 4D/5D.");

    NODE_VALIDATION_CHECK(op,
        grid_shape[last_dim].compatible(data_dims),
        "The last dimension of grid tensor's shape has to be equal to %d.", data_dims);

    NODE_VALIDATION_CHECK(op,
            grid_shape[0] == data_shape[0],
            "The batch dimension in the input data tensor's shape doesn't match the batch dimension in "
            "the grid tensor's shape.");

    auto output_shapes = std::vector<TRShape>(1);
    auto& output_shape = output_shapes.front();
    output_shape.push_back(data_shape[0]);
    output_shape.push_back(data_shape[1]);
    output_shape.push_back(grid_shape[1]);
    output_shape.push_back(grid_shape[2]);
    if (grid_rank == 5)
        output_shape.push_back(grid_shape[3]);
    return output_shapes;
}

}  // namespace v9
}  // namespace op
}  // namespace ov
