//
// Created by finne on 07.04.2023.
//

#pragma once
#include "../../data/matrix_dense.h"
#include "../../data/sarray.h"

namespace operations {

// clang-format off
__global__ void select_bp_kernel(
          float** __restrict__ inputs,
          float*  __restrict__ output,
    const float*  __restrict__ indices,
    unsigned int m,
    unsigned int n,
    unsigned int ld) {
    // clang-format on

    int thread_n = blockIdx.x * blockDim.x + threadIdx.x;
    int thread_z = blockIdx.y * blockDim.y + threadIdx.y;

    if (thread_n >= n)
        return;

    float* input = inputs[thread_z];
    int    idx   = int(indices[thread_n]);

    if (idx == thread_z) {
        for (int i = 0; i < m; i++) {
            input[MATRIX_INDEX(ld, i, thread_n)] = output[MATRIX_INDEX(ld, i, thread_n)];
        }
    } else {
        // TODO: reconsider
        // NOTE: technically correct but in reality not needed
        for (int i = 0; i < m; i++) {
            inputs[MATRIX_INDEX(ld, i, thread_n)] = 0;
        }
    }
}

// clang-format off
void select_bp_host(
          float** __restrict__ inputs,
          float*  __restrict__ output,
    const float*  __restrict__ indices,
    unsigned int heads,
    unsigned int m,
    unsigned int n,
    unsigned int ld) {
    // clang-format on
    for (int x = 0; x < n; x++) {
        size_t input_head = int(indices[x]);
        for (int z = 0; z < heads; z++) {
            float* input = inputs[z];
            for (int y = 0; y < m; y++) {
                if (input_head == z) {
                    input[MATRIX_INDEX(ld, y, x)] = output[MATRIX_INDEX(ld, y, x)];
                } else {
                    input[MATRIX_INDEX(ld, y, x)] = 0;
                }
            }
        }
    }
}

// clang-format off
template<data::Device DEV>
void select_bp(      data::SArray     <float*>& inputs,
               const data::DenseMatrix<float >& output,
               const data::SArray     <float >& indices) {
    // clang-format on
    if constexpr (data::is_cpu(DEV)) {
        // a block has 128 x 1 threads
        dim3 block(128, 1);
        // we need to spawn N threads in the y for N outputs
        dim3 grid(std::ceil((float) output.n / 128.0f), inputs.size());
        select_bp_kernel<<<grid, block>>>(inputs .address<DEV>(),
                                          output .first  <DEV>(),
                                          indices.address<DEV>(),
                                          output.m,
                                          output.n,
                                          output.ld);
    } else {
        select_bp_host(inputs .address<DEV>(),
                       output .first  <DEV>(),
                       indices.address<DEV>(),
                       inputs.size(),
                       output.m,
                       output.n,
                       output.ld);
    }
}

}    // namespace operations

