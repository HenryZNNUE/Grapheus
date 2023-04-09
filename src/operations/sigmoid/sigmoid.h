
#pragma once
#include "../../data/sarray.h"
#include "../../data/device.h"
#include "../../data/matrix_dense.h"

#include <iostream>

namespace operations {
// clang-format off
void sigmoid_host(
    const float* A,
          float* B,
    float scalar,
    size_t m,
    size_t n,
    size_t lda,
    size_t ldb);

__global__ void sigmoid_kernel(
    const float* __restrict__ A,
          float* __restrict__ B,
    float scalar,
    size_t m,
    size_t n,
    size_t lda,
    size_t ldb);

__global__ void sigmoid_kernel_fast(
    const float* __restrict__ A,
          float* __restrict__ B,
    float scalar,
    size_t size);
// clang-format on

// clang-format off
template<data::Device DEV>
inline void sigmoid(const data::DenseMatrix<float> &A,
                          data::DenseMatrix<float> &B,
                    float scalar){

    ASSERT(A.size() == B.size());
    ASSERT(A    .first<DEV>());
    ASSERT(B    .first<DEV>());
    // clang-format on

    if constexpr(data::is_gpu(DEV)){

        // check if any of the two matrices is a submatrix
        if(A.ld != A.m ||
           B.ld != B.m){
            // optimise block sizes depending on output size
            // TODO properly optimise this
            int block_size_x;
            int block_size_y;
            if(A.m > 128){
                block_size_x = 1;
                block_size_y = 32;
            } else if(A.m > 8){
                block_size_x = 32;
                block_size_y = 8;
            } else{
                block_size_x = 512;
                block_size_y = 1;
            };

            dim3 block(block_size_x, block_size_y);
            dim3 grid (std::ceil((float)A.n / block_size_x),
                       std::ceil((float)A.m / block_size_y));
            sigmoid_kernel<<<grid, block>>>(
                A    .first<DEV>(),
                B    .first<DEV>(),
                scalar,
                A.m,
                A.n,
                A.ld,
                B.ld);
        } else{
            // this is the faster method
            // assumes all data is contigious in memory and uses
            // 1d-indexing
            constexpr int block_size = 512;

            dim3 block(block_size);
            dim3 grid (std::ceil((float)A.size() / block_size));
            sigmoid_kernel_fast<<<grid, block>>>(
                A    .first<DEV>(),
                B    .first<DEV>(),
                scalar,
                A.size());
        }
    }else{
        sigmoid_host(
            A    .first<DEV>(),
            B    .first<DEV>(),
            scalar,
            A.m,
            A.n,
            A.ld,
            B.ld);
    }
}

}

// clang-format on
