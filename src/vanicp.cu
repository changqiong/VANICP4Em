/**
   This file is part of VANICP. (https://github.com/changqiong/VANICP4Em.git).

   Copyright (c) 2025 Qiong Chang.

   VANICP is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   any later version.

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
#include "vanicp.h"
#include "utils.cu"
#include <time.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>


#define BLOCK_SIZE 128
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
double dist_GPU(T x1, T y1, T z1,
		T x2, T y2, T z2){
  return static_cast<double>(sqrtf((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2)));
}

/*****************************      Kernel Function         ******************************/
//
// Kernel function to find the nearest neighbor
__global__ void nearest_neighbor_kernel(
					const float*        src,        
					const float*        dst,        
					const unsigned int* mem_addstart_comp_device,  
					const unsigned int* src_int,    
					int                 src_count,
					int                 dst_count,
					int*                best_neighbor, 
					double*             best_dist,     
					int                 iter_num       
					)
{
  // Number of source points processed by each thread
  int num_src_pts_per_thread = (src_count - 1) / (gridDim.x * blockDim.x) + 1;

  double current_best_dist = INF;
  int    current_best_neighbor = 0;
  int    current_index_src = 0;

  float x1, y1, z1, x2, y2, z2;
  unsigned int x1_uint, y1_uint, z1_uint;
  double dist_d;

  // Each thread loops over the subset of source points assigned to it
  for (int j = 0; j < num_src_pts_per_thread; j++) {

    current_index_src =
      blockIdx.x * blockDim.x * num_src_pts_per_thread +
      j * blockDim.x +
      threadIdx.x;

    if (current_index_src < src_count) {

      current_best_dist = INF;
      current_best_neighbor = 0;

      // Load source point (SoA layout)
      x1 = src[current_index_src];
      y1 = src[current_index_src + src_count];
      z1 = src[current_index_src + src_count * 2];

      // Convert to integer voxel indices
      x1_uint = (unsigned int)(fma(x1, SCALE, SCALE));
      y1_uint = (unsigned int)(fma(y1, SCALE, SCALE));
      z1_uint = (unsigned int)(fma(z1, SCALE, SCALE));

      unsigned int index_voxel = 0;
      unsigned int index_addr = 0;

      // Encode voxel index
      index_voxel |= INDEX_VOXEL_HIST_X;
      index_voxel |= INDEX_VOXEL_HIST_Y;
      index_voxel |= INDEX_VOXEL_HIST_Z;

      index_addr = mem_addstart_comp_device[index_voxel];

      int start_index = 1;
      unsigned int checkpoint = src_int[index_addr + start_index];
      unsigned int pointNum   = checkpoint;

      // Check whether the voxel is dilated
      if ((checkpoint & 0xff000000) == 0xff000000) {
	index_voxel = checkpoint & VOXEL_MASK;
	index_addr = mem_addstart_comp_device[index_voxel];
	pointNum = src_int[index_addr + start_index];
      }

      // -------------------------------
      // Case 1: Candidate list exists in this voxel
      // -------------------------------
      if (pointNum != 0) {
	for (int count = 0; count < pointNum; count++) {

	  int dst_idx = src_int[index_addr + count + 2];

	  x2 = dst[dst_idx];
	  y2 = dst[dst_idx + dst_count];
	  z2 = dst[dst_idx + dst_count * 2];

	  dist_d = dist_GPU(x1, y1, z1, x2, y2, z2);

	  if (dist_d < current_best_dist) {
	    current_best_dist = dist_d;
	    current_best_neighbor = dst_idx;
	  }
	}
      }
      // -------------------------------
      // Case 2: Fallback → brute-force over all dst points
      // -------------------------------
      else {
	for (int current_index_dst = 0;
	     current_index_dst < dst_count;
	     current_index_dst++)
	  {
	    x2 = dst[current_index_dst];
	    y2 = dst[current_index_dst + dst_count];
	    z2 = dst[current_index_dst + dst_count * 2];

	    dist_d = dist_GPU(x1, y1, z1, x2, y2, z2);

	    if (dist_d < current_best_dist) {
	      current_best_dist = dist_d;
	      current_best_neighbor = current_index_dst;
	    }
	  }
      }

      // Store results for this source point
      best_dist[current_index_src] = current_best_dist;
      best_neighbor[current_index_src] = current_best_neighbor;
    }
  }
}

// Reorder a 3D point array (SoA: [x | y | z]) according to nearest-neighbor indices.
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

    src_reorder[idx] = src[target_index];

    src_reorder[idx + reorder_y_offset] = src[target_index + src_y_offset];

    src_reorder[idx + reorder_z_offset] = src[target_index + src_z_offset];
  }
}

// GPU kernel for voxelization-based hashing into a histogram
__global__ void _voxelizationHash_gpu_kernel(const float* __restrict__ dst_device,
					     unsigned int* __restrict__ dst_hist_device,
					     int num_points_dst)
{
  int num_points_per_thread = (num_points_dst - 1) / (gridDim.x * blockDim.x) + 1;

  int current_index = 0;

  float x1 = 0.0f, y1 = 0.0f, z1 = 0.0f;
  unsigned int x1_uint = 0, y1_uint = 0, z1_uint = 0;

  for (int j = 0; j < num_points_per_thread; ++j) {
    current_index = blockIdx.x * blockDim.x * num_points_per_thread + j * blockDim.x + threadIdx.x;

    if (current_index < num_points_dst) {
      x1 = dst_device[current_index];
      y1 = dst_device[current_index + num_points_dst];
      z1 = dst_device[current_index + 2 * num_points_dst];

      x1_uint = static_cast<unsigned int>(fma(x1, SCALE, SCALE));
      y1_uint = static_cast<unsigned int>(fma(y1, SCALE, SCALE));
      z1_uint = static_cast<unsigned int>(fma(z1, SCALE, SCALE));

      unsigned int index_block = 0;
      index_block |= INDEX_VOXEL_HIST_X;  
      index_block |= INDEX_VOXEL_HIST_Y;  
      index_block |= INDEX_VOXEL_HIST_Z;  

      atomicAdd(&dst_hist_device[index_block], 1);
    }
  }
}

// GPU voxelization kernel: assign each input point to a voxel bin
__global__ void _voxelization_comp_gpu_kernel(const float*	__restrict__ dst_device, unsigned int*	__restrict__ dst_int_comp_device,	unsigned int*	__restrict__ mem_addstart_comp_device,	int	num_points_dst)
{
  // Number of points handled by each thread
  int num_points_per_thread = (num_points_dst - 1) / (gridDim.x * blockDim.x) + 1;

  int current_index = 0;

  float x1 = 0.0f, y1 = 0.0f, z1 = 0.0f;
  unsigned int x1_uint = 0, y1_uint = 0, z1_uint = 0;

  // Loop over all points assigned to this thread
  for (int j = 0; j < num_points_per_thread; ++j) {
    current_index = blockIdx.x * blockDim.x * num_points_per_thread + j * blockDim.x + threadIdx.x;

    if (current_index < num_points_dst) {
      x1 = dst_device[current_index];
      y1 = dst_device[current_index + num_points_dst];
      z1 = dst_device[current_index + 2 * num_points_dst];

      x1_uint = static_cast<unsigned int>(fma(x1, SCALE, SCALE));
      y1_uint = static_cast<unsigned int>(fma(y1, SCALE, SCALE));
      z1_uint = static_cast<unsigned int>(fma(z1, SCALE, SCALE));

      unsigned int index_block = 0;
      index_block |= INDEX_VOXEL_HIST_X;
      index_block |= INDEX_VOXEL_HIST_Y;
      index_block |= INDEX_VOXEL_HIST_Z;

      unsigned int add = mem_addstart_comp_device[index_block];

      bool lock = true;
      while (lock) {
	if (atomicCAS(&dst_int_comp_device[add], 0u, 1u) == 0u) {
	  int index_voxel = dst_int_comp_device[add + 1];
	  dst_int_comp_device[add + index_voxel + 2] = current_index;
	  dst_int_comp_device[add + 1] = index_voxel + 1;
	  atomicExch(&dst_int_comp_device[add], 0u);
	  lock = false;
	}
      }
    }
  }
}


__device__ __forceinline__
void dilate_neighbor(unsigned int* __restrict__ dst_int_comp_device, const unsigned int* __restrict__ mem_addstart_comp_device,	unsigned int neighbor_voxel_index,	unsigned int update_value)
{
  // Get the head address of the neighbor voxel's compressed list
  unsigned int add = mem_addstart_comp_device[neighbor_voxel_index];

  if (atomicCAS(&dst_int_comp_device[add], 0u, 1u) == 0u) {
    if (dst_int_comp_device[add + 1] == 0u) {
      dst_int_comp_device[add + 1] = update_value;
    }
  }
}

__global__ void dilation_comp_core(unsigned int* __restrict__ dst_int_comp_device,	unsigned int* __restrict__ mem_addstart_comp_device, int loop)
{
  // Encode 3D voxel index:
  unsigned int voxel_index = (static_cast<unsigned int>(threadIdx.y) << (VOXEL_BITS_SINGLE * 2)) | (static_cast<unsigned int>(blockIdx.x)  <<  VOXEL_BITS_SINGLE) | static_cast<unsigned int>(blockIdx.y);

  unsigned int index = mem_addstart_comp_device[voxel_index];

  // Current voxel's state/root info
  unsigned int voxel_num_temp = dst_int_comp_device[index + 1];

  if (voxel_num_temp != 0u) {
    unsigned int root_index  = (loop == 0) ? voxel_index : (voxel_num_temp & VOXEL_MASK);
    unsigned int update_value = (0xff000000u | root_index);

    // -------------------- Neighbor in -Y direction --------------------
    if (threadIdx.x == 0 && threadIdx.y > 0) {
      unsigned int i_temp = static_cast<unsigned int>(threadIdx.y - 1);
      unsigned int t =
	(i_temp << (VOXEL_BITS_SINGLE * 2)) |
	(voxel_index & CURRENT_INDEX_MASK_X);

      dilate_neighbor(dst_int_comp_device,
		      mem_addstart_comp_device,
		      t,
		      update_value);
    }

    // -------------------- Neighbor in +Y direction --------------------
    if (threadIdx.x == 1 && threadIdx.y < VOXEL_NUM_SINGLE - 1) {
      unsigned int i_temp = static_cast<unsigned int>(threadIdx.y + 1);
      unsigned int t =
	(i_temp << (VOXEL_BITS_SINGLE * 2)) |
	(voxel_index & CURRENT_INDEX_MASK_X);

      dilate_neighbor(dst_int_comp_device,
		      mem_addstart_comp_device,
		      t,
		      update_value);
    }

    // -------------------- Neighbor in -X direction --------------------
    if (threadIdx.x == 2 && blockIdx.x > 0) {
      unsigned int i_temp = static_cast<unsigned int>(blockIdx.x - 1);
      unsigned int t =
	(i_temp << VOXEL_BITS_SINGLE) |
	(voxel_index & CURRENT_INDEX_MASK_Y);

      dilate_neighbor(dst_int_comp_device,
		      mem_addstart_comp_device,
		      t,
		      update_value);
    }

    // -------------------- Neighbor in +X direction --------------------
    if (threadIdx.x == 3 && blockIdx.x < VOXEL_NUM_SINGLE - 1) {
      unsigned int i_temp = static_cast<unsigned int>(blockIdx.x + 1);
      unsigned int t =
	(i_temp << VOXEL_BITS_SINGLE) |
	(voxel_index & CURRENT_INDEX_MASK_Y);

      dilate_neighbor(dst_int_comp_device,
		      mem_addstart_comp_device,
		      t,
		      update_value);
    }

    // -------------------- Neighbor in -Z direction --------------------
    if (threadIdx.x == 4 && blockIdx.y > 0) {
      unsigned int i_temp = static_cast<unsigned int>(blockIdx.y - 1);
      unsigned int t =
	i_temp |
	(voxel_index & CURRENT_INDEX_MASK_Z);

      dilate_neighbor(dst_int_comp_device,
		      mem_addstart_comp_device,
		      t,
		      update_value);
    }

    // -------------------- Neighbor in +Z direction --------------------
    if (threadIdx.x == 5 && blockIdx.y < VOXEL_NUM_SINGLE - 1) {
      unsigned int i_temp = static_cast<unsigned int>(blockIdx.y + 1);
      unsigned int t =
	i_temp |
	(voxel_index & CURRENT_INDEX_MASK_Z);

      dilate_neighbor(dst_int_comp_device,
		      mem_addstart_comp_device,
		      t,
		      update_value);
    }

    atomicExch(&dst_int_comp_device[index], 0u);
  }

  __syncthreads();
}


__host__ void _compute_optimal_rigid_transform_SVD(
						   cublasHandle_t       blas_handle,
						   cusolverDnHandle_t   solver_handle,
						   const float*         src_zm_device,       
						   const float*         dst_zm_reordered_device,
						   const float*         sum_device_src,     
						   const float*         sum_device_dst,     
						   int                  num_data_pts,
						   double*              trans_matrix_device  
						   ){
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

  check_return_status(cublasSgemm(
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
  double* d_U  = nullptr; 
  double* d_Vt = nullptr; 
  double* d_S  = nullptr; 

  check_return_status(cudaMalloc(
				 reinterpret_cast<void**>(&d_U),
				 Nrows * Nrows * sizeof(double)));
  check_return_status(cudaMalloc(
				 reinterpret_cast<void**>(&d_Vt),
				 Ncols * Ncols * sizeof(double)));
  check_return_status(cudaMalloc(
				 reinterpret_cast<void**>(&d_S),
				 Nrows * sizeof(double))); 

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
  check_return_status(cusolverDnDgesvd(
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
  check_return_status(cudaMalloc(reinterpret_cast<void**>(&rot_matrix_device),
				 3 * 3 * sizeof(double)));

  m   = 3;
  k   = 3;
  n   = 3;
  lda = k;
  ldb = k;
  ldc = m;

  check_return_status(cublasDgemm(
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

  double* sum_src_d = nullptr;
  check_return_status(cudaMalloc(
				 reinterpret_cast<void**>(&sum_src_d),
				 3 * sizeof(double)));

  cast_float_to_double<<<1, 1>>>(sum_device_src, sum_src_d, 3);

  m   = 3;
  k   = 3;
  n   = 1;
  lda = m;
  ldb = k;
  ldc = m;

  check_return_status(cublasDgemm(
				  blas_handle,
				  CUBLAS_OP_N, CUBLAS_OP_N,
				  m, n, k,
				  alpha_d_ptr,
				  rot_matrix_device, lda,
				  sum_src_d,         ldb,
				  beta_d_ptr,
				  t_device,          ldc));

  check_return_status(cudaFree(sum_src_d));

  const double minus_one = -1.0;
  check_return_status(
		      cublasDscal(
				  blas_handle,
				  3,
				  &minus_one,
				  t_device,
				  1));

  double* sum_dst_d = nullptr;
  check_return_status(cudaMalloc(reinterpret_cast<void**>(&sum_dst_d),
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
  double one_val = 1.0;
  check_return_status(
		      cublasSetVector(
				      1,
				      sizeof(double),
				      &one_val,
				      1,
				      trans_matrix_device + 15,
				      1));

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
				 const float*   point_array_device,
				 const float*   ones_device,      
				 int            num_data_pts,
				 float*         point_array_zm_device, 
				 float*         sum_device_dst     
				 ){

  const float alpha = 1.0f;
  const float beta  = 0.0f;
  int m = 1;
  int k = num_data_pts;
  int n = 3;
  int lda = m;
  int ldb = k;
  int ldc = m;

  check_return_status(cublasSgemm(
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
  const float inv_num = 1.0f / static_cast<float>(num_data_pts);
  for (int i = 0; i < 3; ++i) {
    avg_host[i] *= inv_num;
  }

  check_return_status(
		      cudaMemcpy(
				 point_array_zm_device,
				 point_array_device,
				 3 * num_data_pts * sizeof(float),
				 cudaMemcpyDeviceToDevice
				 )
		      );

  for (int i = 0; i < 3; ++i) {
    const float neg_mean = -avg_host[i];
    check_return_status(cublasSaxpy(
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
				       unsigned int*      mem_addstart_comp_device,
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
					   mem_addstart_comp_device,
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
				   float*                      src_4d_device
				   )
{
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


__global__ void map_hist(uint* hist, uint* tmp){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < VOXEL_NUM) {
    uint c = hist[idx];
    tmp[idx] = (c == 0 ? 2 : c + 2);
  }
}

__host__ unsigned int _hist_asyc(const float *dst_device, unsigned int *dst_hist_device, int num_data_pts_dst){
  cudaEvent_t start, stop;
  CLOCK_START(start,stop);
  
  dim3 fullGrids((num_data_pts_dst + BLOCK_SIZE - 1) / BLOCK_SIZE);
  _voxelizationHash_gpu_kernel<<<fullGrids, BLOCK_SIZE>>>(dst_device, dst_hist_device, num_data_pts_dst);
  cudaDeviceSynchronize();
  std::string Output_message ("histogram Time: ");
  CLOCK_END(start,stop);
  
  uint current_point_count=0;
  uint voxel_index = 0;
  for(int count = 0; count<VOXEL_NUM; count++){
    current_point_count = dst_hist_device[count]==0?2:dst_hist_device[count]+2;
    dst_hist_device[count] = voxel_index;
    voxel_index += current_point_count;
  }

  return voxel_index;

}

__host__ void _filling_comp_gpu(const float *dst_device, unsigned int *dst_int_comp_device, unsigned int *mem_addstart_comp_device, int num_data_pts_dst){
 
  cudaEvent_t start, stop;
  CLOCK_START(start,stop);
  
  dim3 fullGrids((num_data_pts_dst + BLOCK_SIZE - 1) / BLOCK_SIZE);
  _voxelization_comp_gpu_kernel<<<fullGrids, BLOCK_SIZE>>>(dst_device, dst_int_comp_device, mem_addstart_comp_device, num_data_pts_dst);

  cudaDeviceSynchronize();
  std::string Output_message ("GPU Voxelization Running Time: ");
  CLOCK_END(start,stop);
  
  CLOCK_START(start,stop);

  dim3 VoxelGrids(VOXEL_NUM_SINGLE ,VOXEL_NUM_SINGLE);
  dim3 th(6,VOXEL_NUM_SINGLE);

  for(int loop = 0; loop < LOOP_NUM; loop++)
    dilation_comp_core<<<VoxelGrids, th>>>(dst_int_comp_device, mem_addstart_comp_device, loop);
  cudaDeviceSynchronize();
  Output_message =("GPU Dilation Filling Running Time: ");
  CLOCK_END(start,stop);
}

/***************************    Main  algorithm *********************************/
__host__ int vanicp(
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
  std::vector<float> average_host(3, 0.0f); 

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

  unsigned int *mem_int_comp_device = nullptr;
  unsigned int *mem_addstart_comp_device = nullptr;
  unsigned int *mem_hist_device = nullptr;

  // -----------------------------------
  // CUBLAS / CUSOLVER initialization
  // -----------------------------------
  cublasHandle_t    handle;
  cusolverDnHandle_t solver_handle;
  cublasCreate(&handle);
  cusolverDnCreate(&solver_handle);

  check_return_status(cudaMallocManaged((void**)&mem_hist_device, VOXEL_NUM*sizeof(unsigned int)));
  check_return_status(cudaMalloc((void**)&mem_addstart_comp_device, VOXEL_NUM*sizeof(unsigned int)));
  cudaMemset(mem_hist_device, 0, VOXEL_NUM*sizeof(unsigned int)); 
  cudaMemset(best_dist_device, 0.0, num_data_pts*sizeof(double)); 

   

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
  check_return_status(
		      cudaMemcpy(src_4d_device, src_device,
				 3 * num_data_pts * sizeof(float),
				 cudaMemcpyDeviceToDevice));

  check_return_status(
		      cudaMemcpy(src_4d_device + 3 * num_data_pts,
				 ones_device,
				 num_data_pts * sizeof(float),
				 cudaMemcpyDeviceToDevice));

  check_return_status(cublasSetVector(num_data_pts, sizeof(float), ones_host.data(), 1, ones_device, 1));
  check_return_status(cudaMemcpy(src_4d_device, src_device, 3 * num_data_pts * sizeof(float), cudaMemcpyDeviceToDevice));
  check_return_status(cudaMemcpy(src_4d_device + 3 * num_data_pts,
				 ones_device, num_data_pts * sizeof(float), cudaMemcpyDeviceToDevice));


  // -----------------------------------
  // ICP main loop
  // -----------------------------------
  double prev_error = 0.0;
  double mean_error = 0.0;
  unsigned int mem_size = 0; 
  mem_size=_hist_asyc(dst_device, mem_hist_device, num_data_pts_dst)*1.5;
  printf("memsize: %d\n", mem_size);
  cudaDeviceSynchronize();

  
  check_return_status(cudaMemcpy(mem_addstart_comp_device, mem_hist_device, VOXEL_NUM * sizeof(unsigned int), cudaMemcpyHostToDevice));
  check_return_status(cudaMalloc((void**)&mem_int_comp_device, mem_size * sizeof(unsigned int)));
  cudaMemset(mem_int_comp_device, 0, mem_size*sizeof(unsigned int));

  _filling_comp_gpu(dst_device, mem_int_comp_device, mem_addstart_comp_device, num_data_pts_dst);
  
  cudaDeviceSynchronize();
  _nearest_neighbor_search(src_4d_device, dst_device,
			   mem_addstart_comp_device, mem_int_comp_device, 
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
			 src_4d_t_device, src_4d_device
			 );

    _nearest_neighbor_search(
			     src_4d_device, dst_device, 
			     mem_addstart_comp_device, mem_int_comp_device,
			     num_data_pts, num_data_pts_dst,
			     best_dist_device, neighbor_device, i);

    check_return_status(
			cudaMemcpy(src_device, src_4d_device,
				   3 * num_data_pts * sizeof(float),
				   cudaMemcpyDeviceToDevice));

    check_return_status(
			cublasDasum(handle, num_data_pts, best_dist_device, 1, &mean_error));

    mean_error /= num_data_pts; 

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
  check_return_status(cudaFree(mem_hist_device));
  check_return_status(cudaFree(mem_addstart_comp_device));
  check_return_status(cudaFree(mem_int_comp_device));

  return iter;
}


