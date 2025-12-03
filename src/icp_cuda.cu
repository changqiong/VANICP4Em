/**
   This file is part of VANICP. (https://github.com/changqiong/VANICP4Em.git).

   Copyright (c) 2025 Qiong Chang.

   VANICP is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   VANICP is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with VANICP.  If not, see <http://www.gnu.org/licenses/>.
**/

#include <iostream>
#include <numeric>
#include <cmath>
#include "Eigen/Eigen"
#include <assert.h>
#include <iomanip>
#include <unistd.h>
#include <string>
#include <Eigen/Dense>
#include <deque>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include "icp.h"
#include "utils.cu"


#define BLOCK_SIZE 64
#define GRID_SIZE 128
#define Rbit(x,y)  (x^(1<<y))

/***************************      RunTime Monitor Start           ********************************/
#define CLOCK_START(start, stop) {		\
    cudaEventCreate(&start);			\
    cudaEventCreate(&stop);			\
    cudaEventRecord(start);			\
  }
 

/***************************      RunTime Monitor End           ********************************/
#define CLOCK_END(start, stop) {					\
    cudaEventRecord(stop);						\
    cudaEventSynchronize(stop);						\
    float milliseconds = 0;						\
    cudaEventElapsedTime(&milliseconds, start, stop);			\
    std::cout<<Output_message<< milliseconds << " ms."<<std::endl;	\
  }


/***************************      Device Function           ********************************/

// Calculate distance in GPU
template <typename T>
__device__ __forceinline__
T dist_GPU(T x1, T y1, T z1,
	   T x2, T y2, T z2){
  return static_cast<T>(sqrtf((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2)));
}

/*****************************      Kernel Function         ******************************/
// Kernel function to find the nearest neighbor for each source point
__global__ void nearest_neighbor_kernel(
    const float*        src,          // [3 * src_count], layout: x[0..N-1], y[0..N-1], z[0..N-1]
    const float*        dst,          // [3 * dst_count], same layout
    const unsigned int* src_int,      // voxel / neighbor index structure
    int                 src_count,
    int                 dst_count,
    int*                best_neighbor,
    double*             best_dist,
    int                 iter_num      // currently unused, kept for interface compatibility
)
{
    (void)iter_num;  // silence unused warning if not used

    const int threads_per_grid = gridDim.x * blockDim.x;
    const int num_src_pts_per_thread =
        (src_count + threads_per_grid - 1) / threads_per_grid;

    
    for (int j = 0; j < num_src_pts_per_thread; ++j) {

        const int src_idx =
            blockIdx.x * blockDim.x * num_src_pts_per_thread +
            j * blockDim.x +
            threadIdx.x;

        if (src_idx >= src_count) {
            continue;
        }

        double current_best_dist     = INF;
        int    current_best_neighbor = 0;

        // -------------------------------------------------------
        // Load source point (x1, y1, z1) in float and quantized uint
        // -------------------------------------------------------
        const float x1 = src[src_idx];
        const float y1 = src[src_idx + src_count];
        const float z1 = src[src_idx + src_count * 2];

        const unsigned int x1_uint =
            static_cast<unsigned int>(fma(x1, SCALE, SCALE));
        const unsigned int y1_uint =
            static_cast<unsigned int>(fma(y1, SCALE, SCALE));
        const unsigned int z1_uint =
            static_cast<unsigned int>(fma(z1, SCALE, SCALE));

        // -------------------------------------------------------
        // Compute voxel block index for this source point
        // (uses INDEX_VOXEL_* macros, assumed defined elsewhere)
        // -------------------------------------------------------
        unsigned int index_block = 0;
        index_block |= INDEX_VOXEL_X;
        index_block |= INDEX_VOXEL_Y;
        index_block |= INDEX_VOXEL_Z;

        const int start_index = 1;
        unsigned int index_voxel = src_int[index_block + start_index];

        // Handle redirection via VOXEL_MASK / root voxel
        if ((index_voxel & 0xff000000u) == 0xff000000u) {
            index_block = ( (index_voxel & VOXEL_MASK) <<
                           (MEMORY_BITS - VOXEL_BITS_SINGLE * 3) );
            index_voxel = src_int[index_block + start_index];
        }

        // -------------------------------------------------------
        // Case 1: voxel has candidate neighbors in src_int
        // -------------------------------------------------------
        if (index_voxel != 0u) {
            for (unsigned int count = 0; count < index_voxel; ++count) {

                const unsigned int x2_uint = src_int[index_block + count * 4 + 4];
                const unsigned int y2_uint = src_int[index_block + count * 4 + 5];
                const unsigned int z2_uint = src_int[index_block + count * 4 + 6];

                const unsigned int dist_uint =
                    dist_GPU(x1_uint, y1_uint, z1_uint,
                             x2_uint, y2_uint, z2_uint);

                const double dist_d = static_cast<double>(dist_uint);

                if (dist_d < current_best_dist) {
                    current_best_dist     = dist_d;
                    current_best_neighbor = static_cast<int>(
                        src_int[index_block + count * 4 + 3]);
                }
            }
        }
        // -------------------------------------------------------
        // Case 2: voxel empty → fall back to brute-force search
        // -------------------------------------------------------
        else {
            for (int dst_idx = 0; dst_idx < dst_count; ++dst_idx) {
                const float x2 = dst[dst_idx];
                const float y2 = dst[dst_idx + dst_count];
                const float z2 = dst[dst_idx + dst_count * 2];

                const double dist_d =
                    dist_GPU(x1, y1, z1, x2, y2, z2);

                if (dist_d < current_best_dist) {
                    current_best_dist     = dist_d;
                    current_best_neighbor = dst_idx;
                }
            }
        }

        // -------------------------------------------------------
        // Store results
        // -------------------------------------------------------
        best_dist[src_idx]     = current_best_dist;
        best_neighbor[src_idx] = current_best_neighbor;
    }
}


// Reorder a 3D point array (SoA: [x | y | z]) according to nearest-neighbor indices.
// src:          [ x0 ... xN-1 | y0 ... yN_dst-1 | z0 ... zN_dst-1 ]
// src_reorder:  [ x'_0 ... x'_{N-1} | y'_0 ... | z'_0 ... ]
__global__ void point_array_reorder(const float* __restrict__ src,
                                    const int*   __restrict__ indices,
                                    int                     num_points,
                                    int                     num_points_dst,
                                    float*       __restrict__ src_reorder)
{
    const int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x  * blockDim.x;

    // src uses num_points_dst as stride, src_reorder uses num_points
    const int src_y_offset        = num_points_dst;
    const int src_z_offset        = num_points_dst * 2;
    const int reorder_y_offset    = num_points;
    const int reorder_z_offset    = num_points * 2;

    for (int idx = tid; idx < num_points; idx += stride) {
        const int target_index = indices[idx];

        // x
        src_reorder[idx] = src[target_index];

        // y
        src_reorder[idx + reorder_y_offset] =
            src[target_index + src_y_offset];

        // z
        src_reorder[idx + reorder_z_offset] =
            src[target_index + src_z_offset];
    }
}

// Voxelization GPU kernel
// dst_device: SoA point cloud (x, y, z)
// dst_int_device: compact voxel structure in global memory
__global__ void _voxelization_gpu_kernel(const float* __restrict__ dst_device,
                                         unsigned int*            dst_int_device,
                                         int                      num_points_dst)
{
    const int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x  * blockDim.x;

    const int y_offset = num_points_dst;
    const int z_offset = num_points_dst * 2;

    for (int idx = tid; idx < num_points_dst; idx += stride) {

        const float x1 = dst_device[idx];
        const float y1 = dst_device[idx + y_offset];
        const float z1 = dst_device[idx + z_offset];

        // Quantize to integer grid
        const unsigned int x1_uint =
            static_cast<unsigned int>(fmaf(x1, SCALE, SCALE));
        const unsigned int y1_uint =
            static_cast<unsigned int>(fmaf(y1, SCALE, SCALE));
        const unsigned int z1_uint =
            static_cast<unsigned int>(fmaf(z1, SCALE, SCALE));

        // Encode voxel block index from quantized coordinates
        unsigned int index_block = 0;
        index_block |= INDEX_VOXEL_X;
        index_block |= INDEX_VOXEL_Y;
        index_block |= INDEX_VOXEL_Z;

        bool acquired = false;
        while (!acquired) {
            // Try to lock this voxel entry
            if (atomicCAS(&dst_int_device[index_block], 0u, 1u) == 0u) {

                int index_voxel = dst_int_device[index_block + 1];

                // Layout: [lock_flag, count, ..., <x,y,z,index>...]
                dst_int_device[index_block + index_voxel * 4 + 3] = idx;
                dst_int_device[index_block + index_voxel * 4 + 4] = x1_uint;
                dst_int_device[index_block + index_voxel * 4 + 5] = y1_uint;
                dst_int_device[index_block + index_voxel * 4 + 6] = z1_uint;

                dst_int_device[index_block + 1] = index_voxel + 1;

                // Unlock
                atomicExch(&dst_int_device[index_block], 0u);
                acquired = true;
            }
        }
    }
}

// Inline helper: try to update a neighboring voxel entry
__device__ __forceinline__
void dilation_update_neighbor(
    unsigned int* dst_int_device,
    unsigned int  voxel_index,
    unsigned int  neighbor_block_idx,
    unsigned int  neighbor_shift_bits,
    unsigned int  current_index_mask,
    unsigned int  update_value
)
{
    // Compute linear index t for the neighbor voxel
    const unsigned int t =
        (neighbor_block_idx << neighbor_shift_bits) |
        ((voxel_index & current_index_mask) << (MEMORY_BITS - VOXEL_BITS_SINGLE * 3));

    // If this location was previously zero, mark it and optionally write root info
    if (atomicCAS(&dst_int_device[t], 0u, 1u) == 0u) {
        if (dst_int_device[t + 1] == 0u) {
            dst_int_device[t + 1] = update_value;
        }
    }
}

// 3D dilation kernel over voxel grid
__global__ void _dilation_gpu_kernel(unsigned int* dst_int_device,
                                    bool*         mark,      // currently unused
                                    int           loop)
{
    (void)mark; // silence unused parameter warning if not used elsewhere

    // Encode 3D voxel index from blockIdx
    const unsigned int voxel_index =
        (blockIdx.x << (VOXEL_BITS_SINGLE * 2)) |
        (blockIdx.y <<  VOXEL_BITS_SINGLE)      |
         blockIdx.z;

    // Base index into packed memory
    const unsigned int index =
        (voxel_index << (MEMORY_BITS - VOXEL_BITS_SINGLE * 3));

    // Number of points in this voxel (or encoded info)
    const unsigned int voxel_num_temp = dst_int_device[index + 1];

    __syncthreads();

    if (voxel_num_temp != 0u) {

        // Determine root index for this dilation step
        const unsigned int root_index  =
            (loop == 0) ? voxel_index : (voxel_num_temp & VOXEL_MASK);

        const unsigned int update_value = 0xff000000u | root_index;

        // X- neighbor (thread 0)
        if (threadIdx.x == 0 && blockIdx.x > 0) {
            const unsigned int nx = blockIdx.x - 1;
            dilation_update_neighbor(
                dst_int_device,
                voxel_index,
                nx,
                MEMORY_BITS - VOXEL_BITS_SINGLE,
                CURRENT_INDEX_MASK_X,
                update_value
            );
        }

        // X+ neighbor (thread 1)
        if (threadIdx.x == 1 && blockIdx.x < VOXEL_NUM_SINGLE - 1) {
            const unsigned int nx = blockIdx.x + 1;
            dilation_update_neighbor(
                dst_int_device,
                voxel_index,
                nx,
                MEMORY_BITS - VOXEL_BITS_SINGLE,
                CURRENT_INDEX_MASK_X,
                update_value
            );
        }

        // Y- neighbor (thread 2)
        if (threadIdx.x == 2 && blockIdx.y > 0) {
            const unsigned int ny = blockIdx.y - 1;
            dilation_update_neighbor(
                dst_int_device,
                voxel_index,
                ny,
                MEMORY_BITS - VOXEL_BITS_SINGLE * 2,
                CURRENT_INDEX_MASK_Y,
                update_value
            );
        }

        // Y+ neighbor (thread 3)
        if (threadIdx.x == 3 && blockIdx.y < VOXEL_NUM_SINGLE - 1) {
            const unsigned int ny = blockIdx.y + 1;
            dilation_update_neighbor(
                dst_int_device,
                voxel_index,
                ny,
                MEMORY_BITS - VOXEL_BITS_SINGLE * 2,
                CURRENT_INDEX_MASK_Y,
                update_value
            );
        }

        // Z- neighbor (thread 4)
        if (threadIdx.x == 4 && blockIdx.z > 0) {
            const unsigned int nz = blockIdx.z - 1;
            dilation_update_neighbor(
                dst_int_device,
                voxel_index,
                nz,
                MEMORY_BITS - VOXEL_BITS_SINGLE * 3,
                CURRENT_INDEX_MASK_Z,
                update_value
            );
        }

        // Z+ neighbor (thread 5)
        if (threadIdx.x == 5 && blockIdx.z < VOXEL_NUM_SINGLE - 1) {
            const unsigned int nz = blockIdx.z + 1;
            dilation_update_neighbor(
                dst_int_device,
                voxel_index,
                nz,
                MEMORY_BITS - VOXEL_BITS_SINGLE * 3,
                CURRENT_INDEX_MASK_Z,
                update_value
            );
        }

        // Clear center voxel's occupancy flag (preserve your original semantics)
        atomicExch(&dst_int_device[index], 0u);
    }

    __syncthreads();
}


/*******************************    Helper Function          ***************************/

__host__ void _compute_optimal_rigid_transform_SVD(
    cublasHandle_t       blas_handle,
    cusolverDnHandle_t   solver_handle,
    const float*         src_zm_device,        // 3×N, zero-mean source
    const float*         dst_zm_reordered_device, // 3×N, zero-mean target (reordered)
    const float*         sum_device_src,       // 3×1, sum of original src
    const float*         sum_device_dst,       // 3×1, sum of original dst
    int                  num_data_pts,
    double*              trans_matrix_device   // 4×4, output transform (double)
)
{
    // ----------------------- 1. Compute 3×3 cross-covariance H -----------------------
    const float alpha_f = 1.0f;
    const float beta_f  = 0.0f;

    float* H_matrix_device = nullptr; // 3×3
    check_return_status(cudaMalloc(
        reinterpret_cast<void**>(&H_matrix_device),
        3 * 3 * sizeof(float)));

    // H = src_zm^T * dst_zm  (3×N)^T * (3×N) -> 3×3
    int m = 3, k = num_data_pts, n = 3;
    int lda = k, ldb = k, ldc = m;

    check_return_status(
        cublasSgemm(
            blas_handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            m, n, k,
            &alpha_f,
            src_zm_device,         lda,
            dst_zm_reordered_device, ldb,
            &beta_f,
            H_matrix_device,       ldc));

    // ----------------------- 2. SVD: H = U * S * V^T (cuSOLVER, double) -----------------------
    const int Nrows = 3;
    const int Ncols = 3;

    int* devInfo = nullptr;
    check_return_status(cudaMalloc(
        reinterpret_cast<void**>(&devInfo),
        sizeof(int)));

    // Copy H (float) -> d_A (double)
    double* d_A = nullptr;
    check_return_status(cudaMalloc(
        reinterpret_cast<void**>(&d_A),
        Nrows * Ncols * sizeof(double)));

    cast_float_to_double<<<1, 1>>>(H_matrix_device, d_A, Nrows * Ncols);

    // Device-side SVD outputs
    double* d_U  = nullptr; // 3×3
    double* d_Vt = nullptr; // 3×3
    double* d_S  = nullptr; // 3

    check_return_status(cudaMalloc(
        reinterpret_cast<void**>(&d_U),
        Nrows * Nrows * sizeof(double)));
    check_return_status(cudaMalloc(
        reinterpret_cast<void**>(&d_Vt),
        Ncols * Ncols * sizeof(double)));
    check_return_status(cudaMalloc(
        reinterpret_cast<void**>(&d_S),
        Nrows * sizeof(double))); // Nrows == Ncols == 3

    int work_size = 0;
    check_return_status(
        cusolverDnDgesvd_bufferSize(
            solver_handle,
            Nrows, Ncols,
            &work_size));

    double* work = nullptr;
    check_return_status(cudaMalloc(
        reinterpret_cast<void**>(&work),
        work_size * sizeof(double)));

    // Full SVD: U (3×3), Vt (3×3)
    check_return_status(
        cusolverDnDgesvd(
            solver_handle,
            'A', 'A',
            Nrows, Ncols,
            d_A, Nrows,
            d_S,
            d_U,  Nrows,
            d_Vt, Ncols,
            work, work_size,
            nullptr,
            devInfo));

    int devInfo_h = 0;
    check_return_status(
        cudaMemcpy(&devInfo_h, devInfo,
                   sizeof(int), cudaMemcpyDeviceToHost));
    if (devInfo_h != 0) {
        std::cout << "Unsuccessful SVD execution\n";
    }

    // Free temporary SVD buffers we no longer need
    check_return_status(cudaFree(work));
    check_return_status(cudaFree(devInfo));
    check_return_status(cudaFree(H_matrix_device));
    check_return_status(cudaFree(d_A));
    check_return_status(cudaFree(d_S));

    // ----------------------- 3. Compute rotation: R = V * U^T -----------------------
    const double alpha_d = 1.0;
    const double beta_d  = 0.0;
    const double* alpha_d_ptr = &alpha_d;
    const double* beta_d_ptr  = &beta_d;

    double* rot_matrix_device = nullptr; // 3×3
    check_return_status(cudaMalloc(
        reinterpret_cast<void**>(&rot_matrix_device),
        3 * 3 * sizeof(double)));

    m   = 3;
    k   = 3;
    n   = 3;
    lda = k;
    ldb = k;
    ldc = m;

    // rot = Vt^T * U^T   (i.e., R = V * U^T)
    check_return_status(
        cublasDgemm(
            blas_handle,
            CUBLAS_OP_T, CUBLAS_OP_T,
            m, n, k,
            alpha_d_ptr,
            d_Vt, lda,
            d_U,  ldb,
            beta_d_ptr,
            rot_matrix_device, ldc));

    check_return_status(cudaFree(d_Vt));
    check_return_status(cudaFree(d_U));

    // ----------------------- 4. Compute translation: t = (sum_dst - R * sum_src) / N -----------------------
    double* t_device = nullptr;  // 3×1
    check_return_status(cudaMalloc(
        reinterpret_cast<void**>(&t_device),
        3 * sizeof(double)));

    // Convert sum_device_src (float[3]) -> double[3]
    double* sum_src_d = nullptr;
    check_return_status(cudaMalloc(
        reinterpret_cast<void**>(&sum_src_d),
        3 * sizeof(double)));

    cast_float_to_double<<<1, 1>>>(sum_device_src, sum_src_d, 3);

    // t = R * sum_src   (3×3) * (3×1)
    m   = 3;
    k   = 3;
    n   = 1;
    lda = m;
    ldb = k;
    ldc = m;

    check_return_status(
        cublasDgemm(
            blas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            m, n, k,
            alpha_d_ptr,
            rot_matrix_device, lda,
            sum_src_d,         ldb,
            beta_d_ptr,
            t_device,          ldc));

    check_return_status(cudaFree(sum_src_d));

    // t = -t
    const double minus_one = -1.0;
    check_return_status(
        cublasDscal(
            blas_handle,
            3,
            &minus_one,
            t_device,
            1));

    // t += sum_dst (converted to double)
    double* sum_dst_d = nullptr;
    check_return_status(cudaMalloc(
        reinterpret_cast<void**>(&sum_dst_d),
        3 * sizeof(double)));
    cast_float_to_double<<<1, 1>>>(sum_device_dst, sum_dst_d, 3);

    const double plus_one = 1.0;
    check_return_status(
        cublasDaxpy(
            blas_handle,
            3,
            &plus_one,
            sum_dst_d, 1,
            t_device,  1));

    check_return_status(cudaFree(sum_dst_d));

    // t /= num_data_pts
    const double inv_num = 1.0 / static_cast<double>(num_data_pts);
    check_return_status(
        cublasDscal(
            blas_handle,
            3,
            &inv_num,
            t_device,
            1));

    // ----------------------- 5. Assemble 4×4 transform matrix -----------------------
    // Layout: we fill as:
    // [ R00 R01 R02  ?  ]
    // [ R10 R11 R12  ?  ]
    // [ R20 R21 R22  ?  ]
    // [ t0  t1  t2   1  ]
    //
    //  - copy each row of R into trans_matrix_device with stride 4
    //  - copy t into last row (indices 12,13,14)
    //  - set trans_matrix_device[15] = 1

    double one_val = 1.0;
    check_return_status(
        cublasSetVector(
            1,
            sizeof(double),
            &one_val,
            1,
            trans_matrix_device + 15,
            1));

    // Copy R rows into 4×4 (row-major-style layout with stride 4)
    for (int row = 0; row < 3; ++row) {
        check_return_status(
            cublasDcopy(
                blas_handle,
                3,
                rot_matrix_device + row * 3,
                1,
                trans_matrix_device + row * 4,
                1));
    }

    // Copy translation vector
    check_return_status(
        cublasDcopy(
            blas_handle,
            3,
            t_device,
            1,
            trans_matrix_device + 12,
            1));

    // ----------------------- 6. Cleanup -----------------------
    check_return_status(cudaFree(rot_matrix_device));
    check_return_status(cudaFree(t_device));
}

__host__ void zero_center_points(
    cublasHandle_t handle,
    const float*   point_array_device,   // [num_data_pts x 3], column-major
    const float*   ones_device,          // [num_data_pts x 1], all ones
    int            num_data_pts,
    float*         point_array_zm_device, // output: zero-mean points, same shape as input
    float*         sum_device_dst         // workspace: length 3 on device
)
{
    // Compute column sums: sum_device_dst(1x3) = ones^T(1xN) * point_array(Nx3)
    const float alpha = 1.0f;
    const float beta  = 0.0f;

    // Dimensions for cublasSgemm:
    // C = A * B
    // A: (m x k)   = (1 x N) -> ones_device
    // B: (k x n)   = (N x 3) -> point_array_device
    // C: (m x n)   = (1 x 3) -> sum_device_dst
    int m = 1;
    int k = num_data_pts;
    int n = 3;
    int lda = m;
    int ldb = k;
    int ldc = m;

    check_return_status(
        cublasSgemm(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            m, n, k,
            &alpha,
            ones_device,        lda,
            point_array_device, ldb,
            &beta,
            sum_device_dst,     ldc
        )
    );

    // Copy sums back to host
    float avg_host[3];
    check_return_status(
        cublasGetVector(
            3,
            sizeof(float),
            sum_device_dst, 1,
            avg_host,       1
        )
    );

    // Convert sums to means
    const float inv_num = 1.0f / static_cast<float>(num_data_pts);
    for (int i = 0; i < 3; ++i) {
        avg_host[i] *= inv_num;
    }

    // Copy original points to zero-mean buffer
    check_return_status(
        cudaMemcpy(
            point_array_zm_device,
            point_array_device,
            3 * num_data_pts * sizeof(float),
            cudaMemcpyDeviceToDevice
        )
    );

    // For each coordinate dimension, subtract its mean:
    // point_array_zm[:, i] -= avg_host[i]
    for (int i = 0; i < 3; ++i) {
        const float neg_mean = -avg_host[i];
        check_return_status(
            cublasSaxpy(
                handle,
                num_data_pts,
                &neg_mean,
                ones_device, 1,
                point_array_zm_device + i * num_data_pts, 1
            )
        );
    }
}

/****************************    Nearest Neighbor Search *******************************/
__host__ void _nearest_neighbor_search(
    const float*       src_device,
    const float*       dst_device,
    unsigned int*      src_int_device,
    int                num_src_points,
    int                num_dst_points,
    double*            best_dist_device,
    int*               best_neighbor_device,
    int                iter_num
)
{
    const int blocks = (num_src_points + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 grid(blocks);
    dim3 block(BLOCK_SIZE);

    nearest_neighbor_kernel<<<grid, block>>>(
        src_device,
        dst_device,
        src_int_device,
        num_src_points,
        num_dst_points,
        best_neighbor_device,
        best_dist_device,
        iter_num
    );
}

__host__ void _registration_kernel(
				   cublasHandle_t              cublas_handle,
				   cusolverDnHandle_t          cusolver_handle,
				   const float*                dst_device,
				   const float*                src_device,
				   const int*                  neighbor_device,
				   const float*                ones_device,
				   int                         num_data_pts,
				   int                         num_data_pts_dst,
				   float*                      dst_reorder_device,
				   float*                      dst_reorder_zm_device,
				   float*                      src_zm_device,
				   float*                      sum_device_dst,
				   float*                      sum_device_src,
				   float*                      src_4d_t_device,
				   float*                      src_4d_device,
				   int                         max_aa,
				   std::deque<Eigen::MatrixXd>& residual_history
				   )
{
  // residual_history currently unused, keep parameter for interface compatibility
  (void)max_aa;
  (void)residual_history;

   
  // -------------------------------------------------------------------------
  // 1. Reorder dst according to nearest neighbors
  // -------------------------------------------------------------------------
  point_array_reorder<<<GRID_SIZE, BLOCK_SIZE>>>(
						 dst_device,
						 neighbor_device,
						 num_data_pts,
						 num_data_pts_dst,
						 dst_reorder_device
						 );

  // -------------------------------------------------------------------------
  // 2. Zero-center src and reordered dst
  // -------------------------------------------------------------------------
  zero_center_points(
		     cublas_handle,
		     dst_reorder_device,
		     ones_device,
		     num_data_pts,
		     dst_reorder_zm_device,
		     sum_device_dst
		     );

  zero_center_points(
		     cublas_handle,
		     src_device,
		     ones_device,
		     num_data_pts,
		     src_zm_device,
		     sum_device_src
		     );

  // -------------------------------------------------------------------------
  // 3. Compute optimal transform via SVD on GPU
  //    trans_matrix_device is 4x4 double
  // -------------------------------------------------------------------------
  double* trans_matrix_device = nullptr;
  check_return_status(
		      cudaMalloc(reinterpret_cast<void**>(&trans_matrix_device),
				 4 * 4 * sizeof(double))
		      );

  _compute_optimal_rigid_transform_SVD(
		     cublas_handle,
		     cusolver_handle,
		     src_zm_device,
		     dst_reorder_zm_device,
		     sum_device_src,
		     sum_device_dst,
		     num_data_pts,
		     trans_matrix_device
		     );

  // -------------------------------------------------------------------------
  // 4. Convert transform to float and apply: src_4d_t = T * src_4d
  // -------------------------------------------------------------------------
  float* trans_matrix_f_device = nullptr;
  check_return_status(
		      cudaMalloc(reinterpret_cast<void**>(&trans_matrix_f_device),
				 4 * 4 * sizeof(float))
		      );

  cast_double_to_float<<<1, 1>>>(
				 trans_matrix_device,
				 trans_matrix_f_device,
				 16
				 );

  const float alpha = 1.0f;
  const float beta  = 0.0f;

  int m = 4;
  int k = 4;
  int n = num_data_pts;
  int lda = m;
  int ldb = n;
  int ldc = m;

  // src_4d_t_device = T(4x4) * src_4d_device(4xN)^T -> (4xN)
  check_return_status(
		      cublasSgemm(
				  cublas_handle,
				  CUBLAS_OP_N,
				  CUBLAS_OP_T,
				  m, n, k,
				  &alpha,
				  trans_matrix_f_device, lda,
				  src_4d_device,         ldb,
				  &beta,
				  src_4d_t_device,       ldc
				  )
		      );

  // -------------------------------------------------------------------------
  // 5. Transpose result back into src_4d_device as N x 4
  // -------------------------------------------------------------------------
  m   = num_data_pts;
  n   = 4;
  lda = n;
  ldb = n;
  ldc = m;

  check_return_status(
		      cublasSgeam(
				  cublas_handle,
				  CUBLAS_OP_T,
				  CUBLAS_OP_T,
				  m, n,
				  &alpha,
				  src_4d_t_device, lda,
				  &beta,
				  src_4d_t_device, ldb,
				  src_4d_device,   ldc
				  )
		      );

  // -------------------------------------------------------------------------
  // 6. Cleanup
  // -------------------------------------------------------------------------
  check_return_status(cudaFree(trans_matrix_device));
  check_return_status(cudaFree(trans_matrix_f_device));
}


__host__ void _filling_gpu(
			   const float*   dst_device,
			   unsigned int*  dst_int_device,
			   int            num_data_pts_dst
			   )
{
  cudaEvent_t start, stop;
  CLOCK_START(start, stop);

  // -------------------------------------------------------------------------
  // 1. Voxelization
  // -------------------------------------------------------------------------
  dim3 full_grid((num_data_pts_dst + BLOCK_SIZE - 1) / BLOCK_SIZE);
  _voxelization_gpu_kernel<<<full_grid, BLOCK_SIZE>>>(
						      dst_device,
						      dst_int_device,
						      num_data_pts_dst
						      );
  cudaDeviceSynchronize();

  std::string Output_message("GPU Voxelization Running Time: ");
  CLOCK_END(start, stop);

  // -------------------------------------------------------------------------
  // 2. Dilation-based filling
  // -------------------------------------------------------------------------
  bool* mark = nullptr;
  check_return_status(
		      cudaMalloc(reinterpret_cast<void**>(&mark),
				 VOXEL_NUM * sizeof(bool))
		      );

  CLOCK_START(start, stop);

  dim3 voxel_grids(VOXEL_NUM_SINGLE, VOXEL_NUM_SINGLE, VOXEL_NUM_SINGLE);
  dim3 threads_per_block(6, 1);

  for (int loop = 0; loop < LOOP_NUM; ++loop) {
    _dilation_gpu_kernel<<<voxel_grids, threads_per_block>>>(
							     dst_int_device,
							     mark,
							     loop
							     );
  }
  cudaDeviceSynchronize();

  Output_message = "GPU Dilation Filling Running Time: ";
  CLOCK_END(start, stop);

  // Cleanup
  check_return_status(cudaFree(mark));
}


/***************************    Main  algorithm *********************************/
__host__ int icp_cuda(
		      const Eigen::MatrixXf& src,
		      const Eigen::MatrixXf& dst,
		      Eigen::MatrixXf&       src_transformed,
		      NEIGHBOR&           neighbor_out,
		      int                    max_iterations,
		      double                 tolerance)
{
  using Eigen::MatrixXd;

  // -----------------------------------
  // Basic shape checks
  // -----------------------------------
  assert(src_transformed.cols() == 4);
  assert(src_transformed.rows() == src.rows());
  assert(src.cols() == dst.cols());
  assert(dst.cols() == 3);

  const int max_aa = 5;
  std::deque<MatrixXd> residual_history;

  // -----------------------------------
  // Host-side variables
  // -----------------------------------
  const int num_data_pts     = static_cast<int>(src.rows());
  const int num_data_pts_dst = static_cast<int>(dst.rows());

  const float* dst_host = dst.data();
  const float* src_host = src.data();
  float*       gpu_temp_res = src_transformed.data();

  // Host buffers (use std::vector instead of malloc)
  std::vector<int>    best_neighbor_host(num_data_pts);
  std::vector<double> best_dist_host(num_data_pts);

  std::vector<float> ones_host(num_data_pts, 1.0f);
  std::vector<float> average_host(3, 0.0f); // currently unused but kept for compatibility

  // -----------------------------------
  // Device-side variables
  // -----------------------------------
  float *dst_device         = nullptr;
  float *dst_reorder_device = nullptr;
  float *src_device         = nullptr;
  float *src_4d_device      = nullptr;
  float *src_4d_t_device    = nullptr;
  float *dst_reorder_zm_device = nullptr;
  float *src_zm_device         = nullptr;
  float *ones_device           = nullptr;
  float *sum_device_src        = nullptr;
  float *sum_device_dst        = nullptr;

  int*    neighbor_device   = nullptr;
  double* best_dist_device  = nullptr;

  unsigned int* mem_int_device = nullptr;

  // -----------------------------------
  // CUBLAS / CUSOLVER initialization
  // -----------------------------------
  cublasHandle_t    handle;
  cusolverDnHandle_t solver_handle;
  cublasCreate(&handle);
  cusolverDnCreate(&solver_handle);

  check_return_status(
		      cudaMalloc(reinterpret_cast<void**>(&mem_int_device),
				 MEMORY_SIZE * sizeof(unsigned int)));

   

  // -----------------------------------
  // CUDA memory allocations
  // -----------------------------------
  check_return_status(
		      cudaMalloc(reinterpret_cast<void**>(&dst_device),
				 3 * num_data_pts_dst * sizeof(float)));

  check_return_status(
		      cudaMalloc(reinterpret_cast<void**>(&dst_reorder_device),
				 3 * num_data_pts * sizeof(float)));

  check_return_status(
		      cudaMalloc(reinterpret_cast<void**>(&src_device),
				 3 * num_data_pts * sizeof(float)));

  check_return_status(
		      cudaMalloc(reinterpret_cast<void**>(&src_4d_device),
				 4 * num_data_pts * sizeof(float)));

  check_return_status(
		      cudaMalloc(reinterpret_cast<void**>(&src_4d_t_device),
				 4 * num_data_pts * sizeof(float)));

  check_return_status(
		      cudaMalloc(reinterpret_cast<void**>(&neighbor_device),
				 num_data_pts * sizeof(int)));

  check_return_status(
		      cudaMalloc(reinterpret_cast<void**>(&dst_reorder_zm_device),
				 3 * num_data_pts * sizeof(float)));

  check_return_status(
		      cudaMalloc(reinterpret_cast<void**>(&src_zm_device),
				 3 * num_data_pts * sizeof(float)));

  check_return_status(
		      cudaMalloc(reinterpret_cast<void**>(&ones_device),
				 num_data_pts * sizeof(float)));

  check_return_status(
		      cudaMalloc(reinterpret_cast<void**>(&sum_device_src),
				 3 * sizeof(float)));

  check_return_status(
		      cudaMalloc(reinterpret_cast<void**>(&sum_device_dst),
				 3 * sizeof(float)));

  check_return_status(
		      cudaMalloc(reinterpret_cast<void**>(&best_dist_device),
				 num_data_pts * sizeof(double)));

  // -----------------------------------
  // Copy data from host to device
  // -----------------------------------
  check_return_status(
		      cudaMemcpy(dst_device, dst_host,
				 3 * num_data_pts_dst * sizeof(float),
				 cudaMemcpyHostToDevice));

  check_return_status(
		      cudaMemcpy(src_device, src_host,
				 3 * num_data_pts * sizeof(float),
				 cudaMemcpyHostToDevice));

  check_return_status(
		      cublasSetVector(num_data_pts, sizeof(float),
				      ones_host.data(), 1,
				      ones_device, 1));

  // Build homogeneous 4D source: [x y z 1]^T
  check_return_status(
		      cudaMemcpy(src_4d_device, src_device,
				 3 * num_data_pts * sizeof(float),
				 cudaMemcpyDeviceToDevice));

  check_return_status(
		      cudaMemcpy(src_4d_device + 3 * num_data_pts,
				 ones_device,
				 num_data_pts * sizeof(float),
				 cudaMemcpyDeviceToDevice));

  // -----------------------------------
  // ICP main loop
  // -----------------------------------
  double prev_error = 0.0;
  double mean_error = 0.0;

  _filling_gpu(dst_device, mem_int_device, num_data_pts_dst);

  _nearest_neighbor_search(src_4d_device, dst_device,
				mem_int_device,
				num_data_pts, num_data_pts_dst,
				best_dist_device, neighbor_device, 0);

  check_return_status(
		      cublasDasum(handle, num_data_pts, best_dist_device, 1, &prev_error));

  prev_error /= num_data_pts;

  int iter = 0;

  for (int i = 0; i < max_iterations; ++i) {

    _registration_kernel(
			 handle, solver_handle,
			 dst_device, src_device, neighbor_device,
			 ones_device,
			 num_data_pts, num_data_pts_dst,                 // const input
			 dst_reorder_device, dst_reorder_zm_device,
			 src_zm_device, sum_device_dst, sum_device_src,  // temp cache
			 src_4d_t_device, src_4d_device,
			 max_aa, residual_history                        // outputs
			 );

    _nearest_neighbor_search(
				  src_4d_device, dst_device,
				  mem_int_device,
				  num_data_pts, num_data_pts_dst,
				  best_dist_device, neighbor_device, i);

    // Update src_device from updated homogeneous coordinates
    check_return_status(
			cudaMemcpy(src_device, src_4d_device,
				   3 * num_data_pts * sizeof(float),
				   cudaMemcpyDeviceToDevice));

    check_return_status(
			cublasDasum(handle, num_data_pts, best_dist_device, 1, &mean_error));

    mean_error /= num_data_pts;
    mean_error /= SCALE;

    if ((std::abs(prev_error - mean_error) < tolerance) ||
	(i == max_iterations - 1))
      {
	std::cout << "Mean_error: " << mean_error << std::endl;
	break;
      }

    prev_error = mean_error;
    iter       = i + 2;  // keep original logic
  }

  // -----------------------------------
  // Copy final neighbor info back to host
  // -----------------------------------
  check_return_status(
		      cudaMemcpy(best_neighbor_host.data(), neighbor_device,
				 num_data_pts * sizeof(int),
				 cudaMemcpyDeviceToHost));

  check_return_status(
		      cudaMemcpy(best_dist_host.data(), best_dist_device,
				 num_data_pts * sizeof(double),
				 cudaMemcpyDeviceToHost));

  neighbor_out.distances.clear();
  neighbor_out.indices.clear();
  neighbor_out.distances.reserve(num_data_pts);
  neighbor_out.indices.reserve(num_data_pts);

  for (int i = 0; i < num_data_pts; ++i) {
    neighbor_out.distances.push_back(best_dist_host[i]);
    neighbor_out.indices.push_back(best_neighbor_host[i]);
  }

  // -----------------------------------
  // Final transform copy back (4D homogeneous)
  // -----------------------------------
  check_return_status(
		      cudaMemcpy(gpu_temp_res, src_4d_device,
				 4 * num_data_pts * sizeof(float),
				 cudaMemcpyDeviceToHost));

  // -----------------------------------
  // Cleanup
  // -----------------------------------
  cublasDestroy(handle);
  cusolverDnDestroy(solver_handle);

  check_return_status(cudaFree(dst_device));
  check_return_status(cudaFree(src_device));
  check_return_status(cudaFree(dst_reorder_device));
  check_return_status(cudaFree(neighbor_device));
  check_return_status(cudaFree(dst_reorder_zm_device));
  check_return_status(cudaFree(src_zm_device));
  check_return_status(cudaFree(ones_device));
  check_return_status(cudaFree(sum_device_src));
  check_return_status(cudaFree(sum_device_dst));
  check_return_status(cudaFree(best_dist_device));
  check_return_status(cudaFree(src_4d_device));
  check_return_status(cudaFree(src_4d_t_device));
  check_return_status(cudaFree(mem_int_device));

  return iter;
}


