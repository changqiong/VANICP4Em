// Code Credit: Pengfei Li
// Email: pli081@ucr.edu
// All rights reserved

#include <iostream>
#include <numeric>
#include <cmath>
#include "icp_cuda.h"
#include "Eigen/Eigen"
#include <assert.h>
#include <iomanip>
#include <unistd.h>
#include <string>
#include <Eigen/Dense>
#include <deque>
#include <time.h>


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

#include <cublas_v2.h>
#include <cusolverDn.h>

#include "support.cu"

/***************************      Device Function           ********************************/

// Calculate distance in GPU
__device__ double dist_GPU(float x1, float y1, float z1,
			   float x2, float y2, float z2){
  return sqrtf((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));
}

__device__ unsigned int dist_GPU_int(unsigned int x1, unsigned int y1, unsigned int z1, unsigned int x2, unsigned int y2, unsigned int z2 ){
  return (unsigned int)sqrtf((x1-x2)*(x1-x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));
}




/*****************************      Kernel Function         ******************************/
// Kernal function to find the nearest neighbor
__global__ void nearest_neighbor_kernel(const float * src, const float * dst, int src_count, int dst_count, int *best_neighbor, double *best_dist){
  // Kernal function to find the nearest neighbor
  // src: source point cloud array, (num_pts, 3), stored in ColMajor (similar for dst)
  // best_neighbor: best neigbor index in dst point set
  // best_dist    : best neigbor distance from src to dst

  // Dynamic reserve shared mem
  extern __shared__ float shared_mem[];

  int num_dst_pts_per_thread = (dst_count - 1)/(gridDim.x * blockDim.x) + 1;
  int num_src_pts_per_thread = (src_count - 1)/(gridDim.x * blockDim.x) + 1;

  int num_dst_pts_per_block = num_dst_pts_per_thread * blockDim.x;
  int num_src_pts_per_block = num_src_pts_per_thread * blockDim.x;

  float *shared_points = (float *)shared_mem;                // num_dst_pts_per_thread * blockDim.x * 3        
   

  int current_index_dst = 0, current_index_src = 0, current_index_shared = 0;
   
  //Step 0: Initialize variables
   
  for(int j = 0; j < num_src_pts_per_thread; j++){
    current_index_src =  blockIdx.x * blockDim.x * num_src_pts_per_thread + j * blockDim.x + threadIdx.x;
    if (current_index_src < src_count){
      best_dist[current_index_src] = INF;  //INF
      best_neighbor[current_index_src] = 0; //0
    }
           
  }
  //printf("test");
  __syncthreads();

  int num_data_chunk = (src_count - 1)/(num_src_pts_per_thread * blockDim.x) + 1;

  for(int i = 0; i < num_data_chunk; i++){
    //Step 1: Copy part of dst points to shared memory
    for(int j = 0; j < num_dst_pts_per_thread; j++){
      // Memory coalescing index
      current_index_dst = i * num_dst_pts_per_block + j * blockDim.x + threadIdx.x;  // TODO: index rotating
      if (current_index_dst < dst_count){
	//Copy 3d points to shared memory
	for(int k = 0; k<3; k++){
	  current_index_shared = j * blockDim.x + threadIdx.x;
	  shared_points[3*current_index_shared]     = dst[current_index_dst];               // copy dst x
	  shared_points[3*current_index_shared + 1] = dst[current_index_dst + dst_count]; // copy dst y
	  shared_points[3*current_index_shared + 2] = dst[current_index_dst + dst_count*2]; // copy dst z
	}
      }
    }

    __syncthreads();
    float x1, y1, z1;
    float x2, y2, z2;
    double dist;
    //Step 2: find closest point from src to dst shared
    for(int j = 0; j < num_src_pts_per_thread; j++){
      current_index_src = blockIdx.x * num_src_pts_per_block + j * blockDim.x + threadIdx.x;
      if(current_index_src < src_count){
	x1 = src[current_index_src];
	y1 = src[current_index_src + src_count];
	z1 = src[current_index_src + src_count*2];
	//    best_dist[current_index_src] = z1;
	//    best_neighbor[current_index_src] = 10;
	for(int k = 0; k < num_dst_pts_per_block; k++){
	  //current_index_shared = k;
	  x2 = shared_points[3*k];
	  y2 = shared_points[3*k + 1];
	  z2 = shared_points[3*k + 2];

	  dist = dist_GPU(x1, y1, z1, x2, y2, z2);
                 
	  if(dist < best_dist[current_index_src]){
	    best_dist[current_index_src] = dist;
	    current_index_dst = i * blockDim.x * num_dst_pts_per_thread + k;
	    best_neighbor[current_index_src] = current_index_dst;
	  }
	}
      }

    }
  }
   
}



// Kernal function to find the nearest neighbor
__global__ void nearest_neighbor_naive_kernel(const float * src, const float * dst, int src_count, int dst_count, int *best_neighbor, double *best_dist){
  // Kernal function to find the nearest neighbor
  // src: source point cloud array, (num_pts, 3), stored in ColMajor (similar for dst)
  // best_neighbor: best neigbor index in dst point set
  // best_dist    : best neigbor distance from src to dst

  // Dynamic reserve shared mem
  int num_src_pts_per_thread = (src_count - 1)/(gridDim.x * blockDim.x) + 1;

  double current_best_dist = INF;
  int current_best_neighbor = 0;
  int current_index_src = 0;

  float x1, y1, z1;
  float x2, y2, z2;
  double dist;
  for(int j = 0; j < num_src_pts_per_thread; j++){
    current_index_src =  blockIdx.x * blockDim.x * num_src_pts_per_thread + j * blockDim.x + threadIdx.x;
    if (current_index_src < src_count){

      current_best_dist = INF;
      current_best_neighbor = 0;
      x1 = src[current_index_src];
      y1 = src[current_index_src + src_count];
      z1 = src[current_index_src + src_count*2];


      for(int current_index_dst = 0; current_index_dst < dst_count; current_index_dst++){
	x2 = dst[current_index_dst];
	y2 = dst[current_index_dst + dst_count];
	z2 = dst[current_index_dst + dst_count*2];


	dist = dist_GPU(x1, y1, z1, x2, y2, z2);
	if(dist < current_best_dist){
	  current_best_dist = dist;
	  current_best_neighbor = current_index_dst;
	}
      }

      best_dist[current_index_src] = current_best_dist;  //INF
      best_neighbor[current_index_src] = current_best_neighbor; //0
   
    }
  }
}

__global__ void nearest_neighbor_new_kernel(const float * src, const float * dst, unsigned int* src_int, int src_count, int dst_count, int *best_neighbor, double *best_dist, int iter_num){
  int num_src_pts_per_thread = (src_count - 1)/(gridDim.x * blockDim.x) + 1;

  double current_best_dist = INF;
  int current_best_neighbor = 0;
  int current_index_src = 0;

   
  float x1,y1,z1, x2,y2,z2;
  unsigned int x1_uint, y1_uint,z1_uint;
  int dist;
  double dist_d;
  for(int j = 0; j < num_src_pts_per_thread; j++){
    current_index_src =  blockIdx.x * blockDim.x * num_src_pts_per_thread + j * blockDim.x + threadIdx.x;
    if (current_index_src < src_count){

      current_best_dist = INF;
      current_best_neighbor = 0;

      x1 = src[current_index_src];
      y1 = src[current_index_src+src_count];
      z1 = src[current_index_src+src_count*2];
      x1_uint = (unsigned int)(fma(x1,SCALE, SCALE));
      y1_uint = (unsigned int)(fma(y1,SCALE, SCALE));
      z1_uint = (unsigned int)(fma(z1,SCALE, SCALE));

      unsigned int index_block=0;
 
      index_block |= INDEX_VOXEL_X;
      index_block |= INDEX_VOXEL_Y;
      index_block |= INDEX_VOXEL_Z;
     
      int start_index;
#if FILLING == 0
      start_index=0;
#elif FILLING==1
      start_index = 1;
#elif FILLING==2 
      start_index = 0;
#elif FILLING==3
      start_index = 1;
#endif
      unsigned int index_voxel=src_int[index_block+start_index];
      unsigned int index_voxel_filling=src_int[index_block+start_index+1];

      if((index_voxel & 0xff000000) == 0xff000000){
	index_block = ((index_voxel&VOXEL_MASK)<<(MEMORY_BITS-VOXEL_BITS_SINGLE*3));
	index_voxel = src_int[index_block+start_index];
	index_voxel_filling = src_int[index_block+start_index+1];
      }
#if FILLING < 2
      if(index_voxel!=0){
	for(int count = 0; count<index_voxel; count++){
	  x2 = dst[src_int[index_block+count*4+3]];
	  y2 = dst[src_int[index_block+count*4+3]+dst_count];
	  z2 = dst[src_int[index_block+count*4+3]+dst_count*2];

	  dist_d = dist_GPU(x1, y1, z1, x2, y2, z2);
	  if(dist_d < current_best_dist){
	    current_best_dist = dist_d;
	    current_best_neighbor = src_int[index_block+count*4+3];
	  }
	}
      }else{
	for(int current_index_dst = 0; current_index_dst < dst_count; current_index_dst++){
	  x2=dst[current_index_dst];
	  y2=dst[current_index_dst+dst_count];
	  z2=dst[current_index_dst+dst_count*2];
	  dist_d = dist_GPU(x1, y1, z1, x2, y2, z2);
	  if(dist_d < current_best_dist){
	    current_best_dist = dist_d;
	    current_best_neighbor = current_index_dst;
	  }
	}
      }
#else
      if(index_voxel!=0){
      	for(int count = 0; count<index_voxel; count++){
      	  x2_uint = src_int[index_block+count*4+4];
      	  y2_uint = src_int[index_block+count*4+5];
      	  z2_uint = src_int[index_block+count*4+6];

      	  dist = dist_GPU_int(x1_uint, y1_uint, z1_uint, x2_uint, y2_uint, z2_uint);
      	  if(dist < current_best_dist){
      	    current_best_dist = dist;
      	    current_best_neighbor = src_int[index_block+count*4+3];
      	  }

      	}
      }else if(index_voxel_filling!=0){
      	for(int count = 0; count<index_voxel_filling; count++){
      	  x2_uint = src_int[index_block+count*4+4];
      	  y2_uint = src_int[index_block+count*4+5];
      	  z2_uint = src_int[index_block+count*4+6];

      	  dist = dist_GPU_int(x1_uint, y1_uint, z1_uint, x2_uint, y2_uint, z2_uint);
      	  if(dist < current_best_dist){
      	    current_best_dist = dist;
      	    current_best_neighbor = src_int[index_block+count*4+3];
      	  }
      	}
      }else{
      	for(int current_index_dst = 0; current_index_dst < dst_count; current_index_dst++){
      	  x2=dst[current_index_dst];
      	  y2=dst[current_index_dst+dst_count];
      	  z2=dst[current_index_dst+dst_count*2];
      	  dist_d = dist_GPU(x1, y1, z1, x2, y2, z2);
      	  if(dist_d < current_best_dist){
      	    current_best_dist = dist_d;
      	    current_best_neighbor = current_index_dst;
      	  }
      	}
      }
#endif
      best_dist[current_index_src] = current_best_dist;  //INF
      best_neighbor[current_index_src] = current_best_neighbor; //0
   
    }
  }
}


__global__ void nearest_neighbor_comp_kernel(const float * src, const float * dst, unsigned int* mem_addstart_comp_device, unsigned int* src_int, int src_count, int dst_count, int *best_neighbor, double *best_dist, int iter_num){
  int num_src_pts_per_thread = (src_count - 1)/(gridDim.x * blockDim.x) + 1;

  double current_best_dist = INF;
  int current_best_neighbor = 0;
  int current_index_src = 0;

   
  float x1,y1,z1, x2,y2,z2;
  unsigned int x1_uint, y1_uint,z1_uint;
  int dist;
  double dist_d;
  for(int j = 0; j < num_src_pts_per_thread; j++){
    current_index_src =  blockIdx.x * blockDim.x * num_src_pts_per_thread + j * blockDim.x + threadIdx.x;
    if (current_index_src < src_count){

      current_best_dist = INF;
      current_best_neighbor = 0;

      x1 = src[current_index_src];
      y1 = src[current_index_src+src_count];
      z1 = src[current_index_src+src_count*2];
      x1_uint = (unsigned int)(fma(x1,SCALE, SCALE));
      y1_uint = (unsigned int)(fma(y1,SCALE, SCALE));
      z1_uint = (unsigned int)(fma(z1,SCALE, SCALE));

      unsigned int index_addr=0;
      unsigned int index_voxel=0;

      index_voxel |= INDEX_VOXEL_HIST_X;
      index_voxel |= INDEX_VOXEL_HIST_Y;
      index_voxel |= INDEX_VOXEL_HIST_Z;
      index_addr = mem_addstart_comp_device[index_voxel];
     
      int start_index = 1;
      unsigned int checkpoint=src_int[index_addr+start_index];
      unsigned int pointNum=checkpoint;

      if((checkpoint & 0xff000000) == 0xff000000){
	index_voxel = checkpoint & VOXEL_MASK;
	pointNum = src_int[mem_addstart_comp_device[index_voxel]+start_index];
	index_addr = mem_addstart_comp_device[index_voxel];
      }
      if(pointNum !=0){
	for(int count = 0; count<pointNum; count++){
	  x2 = dst[src_int[index_addr+count+2]];
	  y2 = dst[src_int[index_addr+count+2]+dst_count];
	  z2 = dst[src_int[index_addr+count+2]+dst_count*2];

	  dist_d = dist_GPU(x1, y1, z1, x2, y2, z2);
	  if(dist_d < current_best_dist){
	    current_best_dist = dist_d;
	    current_best_neighbor = src_int[index_addr+count+2];
	  }
	}
      }else{
	for(int current_index_dst = 0; current_index_dst < dst_count; current_index_dst++){
	  x2=dst[current_index_dst];
	  y2=dst[current_index_dst+dst_count];
	  z2=dst[current_index_dst+dst_count*2];
	  dist_d = dist_GPU(x1, y1, z1, x2, y2, z2);
	  if(dist_d < current_best_dist){
	    current_best_dist = dist_d;
	    current_best_neighbor = current_index_dst;
	  }
	}
      }
      best_dist[current_index_src] = current_best_dist;  //INF
      best_neighbor[current_index_src] = current_best_neighbor; //0
   
    }
  }
}


// Change point array order given the index array
__global__ void point_array_chorder(const float *src, const int *indices, int num_points, int num_points_dst, float *src_chorder){
  int num_point_per_thread = (num_points - 1)/(gridDim.x * blockDim.x) + 1;
  int current_index = 0;
  int target_index = 0;

  for(int j = 0; j < num_point_per_thread; j++){
    current_index =  blockIdx.x * blockDim.x * num_point_per_thread + j * blockDim.x + threadIdx.x;
    if (current_index < num_points){
      target_index = indices[current_index];
      src_chorder[current_index]                 =  src[target_index];     //x
      src_chorder[current_index + num_points  ]  =  src[target_index + num_points_dst  ];     //y
      src_chorder[current_index + num_points*2]  =  src[target_index + num_points_dst*2];     //z
    }
  }
}


// Voxelization GPU kernel
__global__ void _voxelizationHash_gpu_kernel(const float *dst_device, unsigned int *dst_hist_device, int num_points_dst){
  int num_point_per_thread = (num_points_dst - 1)/(gridDim.x * blockDim.x) + 1;
  int current_index = 0;

  float x1=0.0f, y1=0.0f, z1=0.0f;
  unsigned int x1_uint = 0, y1_uint=0, z1_uint =0;
  for(int j = 0; j < num_point_per_thread; j++){
    current_index =  blockIdx.x * blockDim.x * num_point_per_thread + j * blockDim.x + threadIdx.x;
    if (current_index < num_points_dst){
      x1=dst_device[current_index];
      y1=dst_device[current_index+num_points_dst];
      z1=dst_device[current_index+num_points_dst*2];
      x1_uint = (unsigned int)(fma(x1,SCALE, SCALE));
      y1_uint = (unsigned int)(fma(y1,SCALE, SCALE));
      z1_uint = (unsigned int)(fma(z1,SCALE, SCALE));
      unsigned int index_block=0;
      index_block |= INDEX_VOXEL_HIST_X;
      index_block |= INDEX_VOXEL_HIST_Y;
      index_block |= INDEX_VOXEL_HIST_Z;
      atomicAdd(&dst_hist_device[index_block], 1);
    }
  }
}


// Voxelization GPU kernel
__global__ void _voxelization_comp_gpu_kernel(const float *dst_device, unsigned int *dst_int_comp_device, unsigned int *mem_addstart_comp_device, int num_points_dst){
  int num_point_per_thread = (num_points_dst - 1)/(gridDim.x * blockDim.x) + 1;
  int current_index = 0;

  float x1=0.0f, y1=0.0f, z1=0.0f;
  unsigned int x1_uint = 0, y1_uint=0, z1_uint =0;
  for(int j = 0; j < num_point_per_thread; j++){
    current_index =  blockIdx.x * blockDim.x * num_point_per_thread + j * blockDim.x + threadIdx.x;
    if (current_index < num_points_dst){
      x1=dst_device[current_index];
      y1=dst_device[current_index+num_points_dst];
      z1=dst_device[current_index+num_points_dst*2];
      x1_uint = (unsigned int)(fma(x1,SCALE, SCALE));
      y1_uint = (unsigned int)(fma(y1,SCALE, SCALE));
      z1_uint = (unsigned int)(fma(z1,SCALE, SCALE));
      unsigned int index_block=0;
      index_block |= INDEX_VOXEL_HIST_X;
      index_block |= INDEX_VOXEL_HIST_Y;
      index_block |= INDEX_VOXEL_HIST_Z;
      unsigned int add = mem_addstart_comp_device[index_block];
      bool lock = true;
      while(lock){	
	if(atomicCAS(&dst_int_comp_device[add], 0, 1)==0){
	  int index_voxel = dst_int_comp_device[add+1];
	  dst_int_comp_device[add+index_voxel+2]=current_index;
	  dst_int_comp_device[add+1]=index_voxel+1;
	  atomicExch(&dst_int_comp_device[add], 0);
	  lock=false;
	}
      }    
    }
  }
}


// Voxelization GPU kernel
__global__ void _voxelization_gpu_kernel(const float *dst_device, unsigned int *dst_int_device, int num_points_dst){
  int num_point_per_thread = (num_points_dst - 1)/(gridDim.x * blockDim.x) + 1;
  int current_index = 0;

  float x1=0.0f, y1=0.0f, z1=0.0f;
  unsigned int x1_uint = 0, y1_uint=0, z1_uint =0;
  for(int j = 0; j < num_point_per_thread; j++){
    current_index =  blockIdx.x * blockDim.x * num_point_per_thread + j * blockDim.x + threadIdx.x;
    if (current_index < num_points_dst){
      x1=dst_device[current_index];
      y1=dst_device[current_index+num_points_dst];
      z1=dst_device[current_index+num_points_dst*2];
      x1_uint = (unsigned int)(fma(x1,SCALE, SCALE));
      y1_uint = (unsigned int)(fma(y1,SCALE, SCALE));
      z1_uint = (unsigned int)(fma(z1,SCALE, SCALE));
      unsigned int index_block=0;
      index_block |= INDEX_VOXEL_X;
      index_block |= INDEX_VOXEL_Y;
      index_block |= INDEX_VOXEL_Z;
      bool lock = true;
      while(lock){
	if(atomicCAS(&dst_int_device[index_block], 0, 1)==0){
	  int index_voxel = dst_int_device[index_block+1];
	  dst_int_device[index_block+index_voxel+3]=current_index;
	  dst_int_device[index_block+1]=index_voxel+1;
	  atomicExch(&dst_int_device[index_block], 0);
	  lock=false;
	}
      }
    }
  }
}

// Scaling GPU kernel
__global__ void _scaling_gpu_kernel(const float *dst_device, unsigned int *dst_int_device, int num_points_dst, float scale_factor){
  int num_point_per_thread = (num_points_dst - 1)/(gridDim.x * blockDim.x) + 1;
  int current_index = 0;

  float x1=0.0f, y1=0.0f, z1=0.0f;
  unsigned int x1_uint = 0, y1_uint=0, z1_uint =0;
  unsigned int x1_scale = 0, y1_scale=0, z1_scale =0;
  for(int j = 0; j < num_point_per_thread; j++){
    current_index =  blockIdx.x * blockDim.x * num_point_per_thread + j * blockDim.x + threadIdx.x;
    if (current_index < num_points_dst){
      x1=dst_device[current_index];
      y1=dst_device[current_index+num_points_dst];
      z1=dst_device[current_index+num_points_dst*2];
      x1_uint = (unsigned int)(fma(x1,SCALE, SCALE));
      y1_uint = (unsigned int)(fma(y1,SCALE, SCALE));
      z1_uint = (unsigned int)(fma(z1,SCALE, SCALE));
     
      x1_scale = (unsigned int)(fma(x1, SCALE*scale_factor, SCALE));
      y1_scale = (unsigned int)(fma(y1, SCALE*scale_factor, SCALE));
      z1_scale = (unsigned int)(fma(z1, SCALE*scale_factor, SCALE));

      if(x1_scale>2.0f*SCALE||y1_scale>2.0f*SCALE||z1_scale>2.0f*SCALE)
      	continue;
      else if(x1_scale<0.0f||y1_scale<0.0f||z1_scale<0.0f)
	continue;

      unsigned int index_block=0;
      index_block |= INDEX_VOXEL_SCALE_X ;
      index_block |= INDEX_VOXEL_SCALE_Y ;
      index_block |= INDEX_VOXEL_SCALE_Z ;
 
      if(atomicCAS(&dst_int_device[index_block], 0, 1)==0){
	unsigned int index_voxel = dst_int_device[index_block+1];
	unsigned int index_voxel_filling = dst_int_device[index_block+2];
	if(index_voxel==0){
	  if(index_voxel_filling==0){
	    dst_int_device[index_block+index_voxel_filling*4+3]=current_index;
	    dst_int_device[index_block+index_voxel_filling*4+4]=x1_uint;
	    dst_int_device[index_block+index_voxel_filling*4+5]=y1_uint;
	    dst_int_device[index_block+index_voxel_filling*4+6]=z1_uint;
	    dst_int_device[index_block+2]=index_voxel_filling+1;
	  }
	}
	atomicExch(&dst_int_device[index_block], 0);
      }
     
    }
  }
}


// Dilation GPU Kernel
__global__ void _dilation_comp_gpu_kernel(unsigned int *dst_int_comp_device, unsigned int *mem_addstart_comp_device,  bool* mark, int loop){
  //unsigned int voxel_index = gridDim.z * gridDim.y * blockIdx.x +  blockIdx.y*gridDim.z + blockIdx.z;
 
  unsigned int voxel_index = (blockIdx.x<<(VOXEL_BITS_SINGLE*2))|(blockIdx.y<<VOXEL_BITS_SINGLE)|blockIdx.z;
  unsigned int index = mem_addstart_comp_device[voxel_index];
  unsigned int i_temp;
  unsigned int t;
 
  unsigned int voxel_num_temp = dst_int_comp_device[index+1];
  __syncthreads();
  if(voxel_num_temp!=0){
     
    unsigned int root_index = loop==0? voxel_index:(voxel_num_temp &VOXEL_MASK);
    unsigned int updata_value=(0xff000000|root_index);
    if(threadIdx.x==0&&blockIdx.x>0){
      i_temp = blockIdx.x-1;
      t = (i_temp<<(VOXEL_BITS_SINGLE*2))|((voxel_index&CURRENT_INDEX_MASK_X));
      unsigned int add=mem_addstart_comp_device[t];
      if(atomicCAS(&dst_int_comp_device[add], 0, 1)==0){
	 if(dst_int_comp_device[add+1]==0)
	 dst_int_comp_device[add+1]= updata_value;
      }
    }
    if(threadIdx.x==1&&blockIdx.x<VOXEL_NUM_SINGLE-1){
      i_temp = (blockIdx.x+1);
      t = (i_temp<<(VOXEL_BITS_SINGLE*2))|((voxel_index&CURRENT_INDEX_MASK_X));
      unsigned int add=mem_addstart_comp_device[t];
      if(atomicCAS(&dst_int_comp_device[add], 0, 1)==0){
	if(dst_int_comp_device[add+1]==0)
	  dst_int_comp_device[add+1]= updata_value;
      }
    }
    if(threadIdx.x==2&&blockIdx.y>0){
      i_temp = blockIdx.y-1;
      t = (i_temp<<VOXEL_BITS_SINGLE)|(voxel_index&CURRENT_INDEX_MASK_Y);
      unsigned int add=mem_addstart_comp_device[t];
      if(atomicCAS(&dst_int_comp_device[add], 0, 1)==0){
	if(dst_int_comp_device[add+1]==0)
	  dst_int_comp_device[add+1]= updata_value;
      }
    }
    if(threadIdx.x==3&&blockIdx.y<VOXEL_NUM_SINGLE-1){
      i_temp = (blockIdx.y+1);
      t = (i_temp<<VOXEL_BITS_SINGLE)|(voxel_index&CURRENT_INDEX_MASK_Y);
      unsigned int add=mem_addstart_comp_device[t];
      if(atomicCAS(&dst_int_comp_device[add], 0, 1)==0){
	if(dst_int_comp_device[add+1]==0)
	  dst_int_comp_device[add+1]= updata_value;
      }
    }
    if(threadIdx.x==4&&blockIdx.z>0){
      i_temp = blockIdx.z-1;
      t = i_temp|(voxel_index&CURRENT_INDEX_MASK_Z);
      unsigned int add=mem_addstart_comp_device[t];
      if(atomicCAS(&dst_int_comp_device[add], 0, 1)==0){
	if(dst_int_comp_device[add+1]==0)
	  dst_int_comp_device[add+1]= updata_value;
      }
    }
    if(threadIdx.x==5&&blockIdx.z<VOXEL_NUM_SINGLE-1){
      i_temp = (blockIdx.z+1);
      t = i_temp|(voxel_index&CURRENT_INDEX_MASK_Z);
      unsigned int add=mem_addstart_comp_device[t];
      if(atomicCAS(&dst_int_comp_device[add], 0, 1)==0){
	if(dst_int_comp_device[add+1]==0)
	  dst_int_comp_device[add+1]= updata_value;
      }
    }
    atomicExch(&dst_int_comp_device[index], 0);

  }
  __syncthreads();      

}




// Dilation GPU Kernel
__global__ void _dilation_gpu_kernel(unsigned int *dst_int_device, bool* mark, int loop){
  //unsigned int voxel_index = gridDim.z * gridDim.y * blockIdx.x +  blockIdx.y*gridDim.z + blockIdx.z;
 
  unsigned int voxel_index = (blockIdx.x<<(VOXEL_BITS_SINGLE*2))|(blockIdx.y<<VOXEL_BITS_SINGLE)|blockIdx.z;
  unsigned int index = (voxel_index <<(MEMORY_BITS-VOXEL_BITS_SINGLE*3));
  unsigned int i_temp;
  unsigned int t;
 
  unsigned int voxel_num_temp = dst_int_device[index+1];
  __syncthreads();
  if(voxel_num_temp!=0){
     
    unsigned int root_index = loop==0? voxel_index:(voxel_num_temp &VOXEL_MASK);
    unsigned int updata_value=(0xff000000|root_index);
    if(threadIdx.x==0&&blockIdx.x>0){
      i_temp = blockIdx.x-1;
      t = (i_temp<<(MEMORY_BITS-VOXEL_BITS_SINGLE))|((voxel_index&CURRENT_INDEX_MASK_X)<<(MEMORY_BITS-VOXEL_BITS_SINGLE*3));
      if(atomicCAS(&dst_int_device[t], 0, 1)==0){
	if(dst_int_device[t+1]==0)
	  dst_int_device[t+1]= updata_value;
      }
    }
    if(threadIdx.x==1&&blockIdx.x<VOXEL_NUM_SINGLE-1){
      i_temp = (blockIdx.x+1);
      t = (i_temp<<(MEMORY_BITS-VOXEL_BITS_SINGLE))|((voxel_index&CURRENT_INDEX_MASK_X)<<(MEMORY_BITS-VOXEL_BITS_SINGLE*3));
      if(atomicCAS(&dst_int_device[t], 0, 1)==0){
	if(dst_int_device[t+1]==0)
	  dst_int_device[t+1]= updata_value;
      }
    }
    if(threadIdx.x==2&&blockIdx.y>0){
      i_temp = blockIdx.y-1;
      t = (i_temp<<(MEMORY_BITS-VOXEL_BITS_SINGLE*2))|((voxel_index&CURRENT_INDEX_MASK_Y)<<(MEMORY_BITS-VOXEL_BITS_SINGLE*3));
      if(atomicCAS(&dst_int_device[t], 0, 1)==0){
	if(dst_int_device[t+1]==0)
	  dst_int_device[t+1]= updata_value;
      }
    }
    if(threadIdx.x==3&&blockIdx.y<VOXEL_NUM_SINGLE-1){
      i_temp = (blockIdx.y+1);
      t = (i_temp<<(MEMORY_BITS-VOXEL_BITS_SINGLE*2))|((voxel_index&CURRENT_INDEX_MASK_Y)<<(MEMORY_BITS-VOXEL_BITS_SINGLE*3));
      if(atomicCAS(&dst_int_device[t], 0, 1)==0){
	if(dst_int_device[t+1]==0)
	  dst_int_device[t+1]= updata_value;
      }
    }
    if(threadIdx.x==4&&blockIdx.z>0){
      i_temp = blockIdx.z-1;
      t = (i_temp<<(MEMORY_BITS-VOXEL_BITS_SINGLE*3))|((voxel_index&CURRENT_INDEX_MASK_Z)<<(MEMORY_BITS-VOXEL_BITS_SINGLE*3));
      if(atomicCAS(&dst_int_device[t], 0, 1)==0){
	if(dst_int_device[t+1]==0)
	  dst_int_device[t+1]= updata_value;
      }
    }
    if(threadIdx.x==5&&blockIdx.z<VOXEL_NUM_SINGLE-1){
      i_temp = (blockIdx.z+1);
      t = (i_temp<<(MEMORY_BITS-VOXEL_BITS_SINGLE*3))|((voxel_index&CURRENT_INDEX_MASK_Z)<<(MEMORY_BITS-VOXEL_BITS_SINGLE*3));
      if(atomicCAS(&dst_int_device[t], 0, 1)==0){
	if(dst_int_device[t+1]==0)
	  dst_int_device[t+1]= updata_value;
      }

    }
    atomicExch(&dst_int_device[index], 0);

  }
  __syncthreads();      

}





/*******************************    Helper Function          ***************************/

__host__ void best_trasnform_SVD(cublasHandle_t handle, cusolverDnHandle_t solver_handle, const float *src_zm_device, const float *dst_chorder_zm_device, const float *sum_device_src, const float *sum_device_dst, int num_data_pts, double *trans_matrix_device){

  const float alf = 1;
  const float bet = 0;
  // const float *alpha = &alf;
  // const float *beta = &bet;


  /***********************            Calculate H matrix            **********************************/
  float *H_matrix_device;
  check_return_status(cudaMalloc((void**)&H_matrix_device, 3 * 3 * sizeof(float)));

  // src_zm_device(N,3) dst_chorder_zm_device(N,3)
  // src_zm_device.T  *  dst_chorder_zm_device
  // cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, K, M, &alpha, A, M, A, M, &beta, B, N);
  // A(MxN) K = N  A'(N,M)
  int m = 3, k = num_data_pts, n = 3;
  int lda=k, ldb=k, ldc=m;
  check_return_status(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alf, src_zm_device, lda, dst_chorder_zm_device, ldb, &bet, H_matrix_device, ldc));
  //print_matrix_device<<<1,1>>>(H_matrix_device, 3, 3);
   

  /****************************   SVD decomposition for trans_matrix   *****************************/
  // --- gesvd only supports Nrows >= Ncols
  // --- column major memory ordering

  const int Nrows = 3;
  const int Ncols = 3;

  // --- cuSOLVE input/output parameters/arrays
  int work_size = 0;
  int *devInfo;           check_return_status(cudaMalloc(&devInfo,          sizeof(int)));

  // --- Setting the device matrix and moving the host matrix to the device
  double *d_A;            check_return_status(cudaMalloc(&d_A,      Nrows * Ncols * sizeof(double)));
  cast_float_to_double<<<1,1>>>(H_matrix_device, d_A, Nrows * Ncols);

   
  // --- device side SVD workspace and matrices
  double *d_U;            check_return_status(cudaMalloc(&d_U,  Nrows * Nrows     * sizeof(double)));
  double *d_Vt;            check_return_status(cudaMalloc(&d_Vt,  Ncols * Ncols     * sizeof(double)));
  double *d_S;            check_return_status(cudaMalloc(&d_S,  min(Nrows, Ncols) * sizeof(double)));

  // --- CUDA SVD initialization
  check_return_status(cusolverDnDgesvd_bufferSize(solver_handle, Nrows, Ncols, &work_size));
  double *work;   check_return_status(cudaMalloc(&work, work_size * sizeof(double)));

  // --- CUDA SVD execution
  check_return_status(cusolverDnDgesvd(solver_handle, 'A', 'A', Nrows, Ncols, d_A, Nrows, d_S, d_U, Nrows, d_Vt, Ncols, work, work_size, NULL, devInfo));
  int devInfo_h = 0;  check_return_status(cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
  if (devInfo_h != 0) std::cout   << "Unsuccessful SVD execution\n\n";
  check_return_status(cudaFree(work));
  check_return_status(cudaFree(devInfo));

  check_return_status(cudaFree(H_matrix_device));
  check_return_status(cudaFree(d_A));
  check_return_status(cudaFree(d_S));

  /**************************      calculating rotation matrix       ******************************/
  const double alfd = 1;
  const double betd = 0;
  const double *alphad = &alfd;
  const double *betad = &betd;

  double *rot_matrix_device;
  check_return_status(cudaMalloc((void**)&rot_matrix_device, 3 * 3 * sizeof(double)));

  m = 3; k = 3; n = 3;
  lda=k; ldb=k; ldc=m;
  // Vt.transpose()*U.transpose();
  check_return_status(cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, m, n, k, alphad, d_Vt, lda, d_U, ldb, betad, rot_matrix_device, ldc));
  check_return_status(cudaFree(d_Vt));
  check_return_status(cudaFree(d_U));

  /***************************      calculating translation matrix    ******************************/
  double *t_matrix_device;
  check_return_status(cudaMalloc((void**)&t_matrix_device, 3 * sizeof(double)));

  m = 3; k = 3; n = 1; //(m,k), (k,n)  -> (m, n)
  lda=m; ldb=k; ldc=m;
  double *sum_device_src_d;            check_return_status(cudaMalloc(&sum_device_src_d, 3 * sizeof(double)));
  cast_float_to_double<<<1,1>>>(sum_device_src, sum_device_src_d, 3);
  check_return_status(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alphad, rot_matrix_device, lda, sum_device_src_d, ldb, betad, t_matrix_device, ldc));
  check_return_status(cudaFree(sum_device_src_d));

  const double scale_trans = -1;
  check_return_status(cublasDscal(handle, 3, &scale_trans, t_matrix_device, 1));

  double *sum_device_dst_d;            check_return_status(cudaMalloc(&sum_device_dst_d, 3 * sizeof(double)));
  cast_float_to_double<<<1,1>>>(sum_device_dst, sum_device_dst_d, 3);
  const double scale_trans_1 = 1;
  check_return_status(cublasDaxpy(handle, 3, &scale_trans_1, sum_device_dst_d, 1, t_matrix_device, 1));
  check_return_status(cudaFree(sum_device_dst_d));

  const double avg_trans = 1/(1.0*num_data_pts);
  check_return_status(cublasDscal(handle, 3, &avg_trans, t_matrix_device, 1));

  /*************         final transformation         ********************/
  // Set the last value to one
  double temp_one = 1;
  check_return_status(cublasSetVector(1, sizeof(double), &temp_one, 1, trans_matrix_device + 15, 1));
  for( int i = 0; i< 3; i++){
    check_return_status(cublasDcopy(handle, 3, rot_matrix_device + i * 3, 1, trans_matrix_device + i * 4, 1));
  }
  check_return_status(cublasDcopy(handle, 3, t_matrix_device, 1, trans_matrix_device + 12, 1));
  check_return_status(cudaFree(rot_matrix_device));
  check_return_status(cudaFree(t_matrix_device));

}

__host__ void zero_center_points(cublasHandle_t handle, const float *point_array_device, const float *ones_device, int num_data_pts, float *point_array_zm_device, float *sum_device_dst){

  const float alf = 1;
  const float bet = 0;
  // const float *alpha = &alf;
  // const float *beta = &bet;

  float *average_host = (float *)malloc(3*sizeof(float));

  /*******************************  zero center dst point array    *****************************************/
  // Do the actual multiplication
  // op ( A ) m × k , op ( B ) k × n and C m × n ,
  // cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  int m = 1, k = num_data_pts, n = 3;
  int lda=m,ldb=k,ldc=m;
  check_return_status(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alf, ones_device, lda, point_array_device, ldb, &bet, sum_device_dst, ldc));
   
  cublasGetVector(3, sizeof(float), sum_device_dst, 1, average_host, 1);
   
  for(int i = 0; i < 3; i++)  average_host[i] /= num_data_pts;
   
   
  check_return_status(cudaMemcpy(point_array_zm_device, point_array_device, 3 * num_data_pts * sizeof(float), cudaMemcpyDeviceToDevice));
   
  for(int i = 0; i < 3; i++)
    {
      const float avg = -average_host[i];
      check_return_status(cublasSaxpy(handle, num_data_pts, &avg, ones_device, 1, point_array_zm_device + i*num_data_pts, 1));
    }

}


/****************************     Warper function *******************************/
// To simplify the function, warper function assumes every device variable is correctly allocated and initialized
// Don't use this unless you are certain about that

__host__ void _nearest_neighbor_cuda_warper(const float *src_device, const float *dst_device, unsigned int* mem_addstart_comp_device, unsigned int * src_int_device, int row_src, int row_dst, double *best_dist_device, int *best_neighbor_device, int iter_num){


#ifndef NN_OPTIMIZE
  nearest_neighbor_naive_kernel<<<GRID_SIZE, BLOCK_SIZE >>>(src_device, dst_device, row_src, row_dst, best_neighbor_device, best_dist_device);

#elif NN_OPTIMIZE == 0
   
  int num_dst_pts_per_thread = (row_dst - 1)/(GRID_SIZE * BLOCK_SIZE) + 1;
  int dyn_size_1 = num_dst_pts_per_thread * BLOCK_SIZE * 3 * sizeof(float);  // memory reserved for shared_points
  nearest_neighbor_kernel<<<GRID_SIZE, BLOCK_SIZE, (dyn_size_1) >>>(src_device, dst_device, row_src, row_dst, best_neighbor_device, best_dist_device);

#elif NN_OPTIMIZE == 1
 
  dim3 fullGrids((row_src + BLOCK_SIZE - 1) / BLOCK_SIZE);
  nearest_neighbor_naive_kernel<<<fullGrids, BLOCK_SIZE >>>(src_device, dst_device, row_src, row_dst, best_neighbor_device, best_dist_device);

#elif NN_OPTIMIZE == 2

   
  dim3 fullGrids((row_src + BLOCK_SIZE - 1) / BLOCK_SIZE);
  //nearest_neighbor_new_kernel<<<fullGrids, BLOCK_SIZE >>>(src_device, dst_device, src_int_device, row_src, row_dst, best_neighbor_device, best_dist_device, iter_num);
  nearest_neighbor_comp_kernel<<<fullGrids, BLOCK_SIZE >>>(src_device, dst_device, mem_addstart_comp_device, src_int_device, row_src, row_dst, best_neighbor_device, best_dist_device, iter_num);
#endif
}

__host__ void _apply_optimal_transform_cuda_warper(cublasHandle_t handle, cusolverDnHandle_t solver_handle, const float *dst_device, const float *src_device, const int *neighbor_device, const float *ones_device, int num_data_pts, int num_data_pts_dst,
						   float *dst_chorder_device, float *dst_chorder_zm_device, float *src_zm_device, float *sum_device_dst, float *sum_device_src,
						   float *src_4d_t_device, float *src_4d_device, std::deque<Eigen::MatrixXd> &residual_history 
						   ){
  cudaEvent_t start, stop;
    CLOCK_START(start,stop);
   
  /*****************************   change order based on the nearest neighbor ******************************/
  point_array_chorder<<<GRID_SIZE, BLOCK_SIZE>>>(dst_device, neighbor_device, num_data_pts, num_data_pts_dst, dst_chorder_device);
       
  /******************************   Calculate Transformation with SVD    ************************************/
  zero_center_points(handle, dst_chorder_device, ones_device, num_data_pts, dst_chorder_zm_device, sum_device_dst);
  zero_center_points(handle, src_device, ones_device, num_data_pts, src_zm_device, sum_device_src);
   
  double *trans_matrix_device; //matrix size is (4,4)
  check_return_status(cudaMalloc((void**)&trans_matrix_device, 4 * 4 * sizeof(double)));
   
  best_trasnform_SVD(handle, solver_handle, src_zm_device, dst_chorder_zm_device, sum_device_src, sum_device_dst, num_data_pts, trans_matrix_device);
  std::string Output_message ("SVD Running Time: ");
    CLOCK_END(start,stop);
    // cudaEventDestroy(start);
    // cudaEventDestroy(stop);

    CLOCK_START(start,stop);
   
  /********************************       Apply transformation       **************************************/
  // Convert to float data
  float *trans_matrix_f_device; //matrix size is (4,4)

  check_return_status(cudaMalloc((void**)&trans_matrix_f_device, 4 * 4 * sizeof(float)));
  cast_double_to_float<<<1,1>>>(trans_matrix_device, trans_matrix_f_device, 16);

#if MATRIX_VISIABLE
  float *trans_matrix_f_host =(float *)malloc(4*4*sizeof(float));
  check_return_status(cudaMemcpy(trans_matrix_f_host, trans_matrix_f_device, 4*4*sizeof(float), cudaMemcpyDeviceToHost));


  std::cout<<std::endl<<"transformation matrix."<<std::endl;
  for(int i = 0 ; i < 4; i++){
    for(int j = 0; j<4; j++){
      printf("%f,", trans_matrix_f_host[i*4+j]);
    }
    printf("\n");
  }
  std::cout<<std::endl;
#endif
  // Matrix multiplication
  const float alf = 1;
  const float bet = 0;
  int m = 4, k = 4, n = num_data_pts;
  int lda=m,ldb=n,ldc=m;
  check_return_status(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, &alf, trans_matrix_f_device, lda, src_4d_device, ldb, &bet, src_4d_t_device, ldc));

  Output_message =("Transformation Running Time: ");
  CLOCK_END(start,stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

   
  /*******************************      Transpose the matrix       *****************************************/
  m = num_data_pts; n = 4;
  lda=n,ldb=n,ldc=m;
  check_return_status(cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, m, n,
				  &alf, src_4d_t_device, lda,
				  &bet, src_4d_t_device, ldb,
				  src_4d_device, ldc));


   
  check_return_status(cudaFree(trans_matrix_device));
  check_return_status(cudaFree(trans_matrix_f_device));
   
}


__host__ unsigned int _hist_asyc(const float *dst_device, unsigned int *dst_hist_device, int num_data_pts_dst){
  cudaEvent_t start, stop;
  CLOCK_START(start,stop);
  
  dim3 fullGrids((num_data_pts_dst + BLOCK_SIZE - 1) / BLOCK_SIZE);
  _voxelizationHash_gpu_kernel<<<fullGrids, BLOCK_SIZE>>>(dst_device, dst_hist_device, num_data_pts_dst);
  cudaDeviceSynchronize();
  
  uint current_point_count=0;
  uint voxel_index = 0;
  for(int count = 0; count<VOXEL_NUM; count++){
    current_point_count = dst_hist_device[count]==0?2:dst_hist_device[count]+2;
    dst_hist_device[count] = voxel_index;
    voxel_index += current_point_count;
  }
  std::string Output_message ("Histgram Time: ");
  CLOCK_END(start,stop);
  
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
  
  bool *mark;
  check_return_status(cudaMalloc((void**)&mark, VOXEL_NUM * sizeof(bool)));

  CLOCK_START(start,stop);

  dim3 VoxelGrids(VOXEL_NUM_SINGLE ,VOXEL_NUM_SINGLE, VOXEL_NUM_SINGLE);
  dim3 th(6,1);

  for(int loop = 0; loop < LOOP_NUM; loop++)
    _dilation_comp_gpu_kernel<<<VoxelGrids, th>>>(dst_int_comp_device, mem_addstart_comp_device, mark, loop);
  cudaDeviceSynchronize();
  Output_message =("GPU Dilation Filling Running Time: ");
  CLOCK_END(start,stop);

}

__host__ void _filling_gpu(const float *dst_device, unsigned int *dst_int_device, int num_data_pts_dst){
 
  cudaEvent_t start, stop;
  CLOCK_START(start,stop);
  
  dim3 fullGrids((num_data_pts_dst + BLOCK_SIZE - 1) / BLOCK_SIZE);
  _voxelization_gpu_kernel<<<fullGrids, BLOCK_SIZE>>>(dst_device, dst_int_device, num_data_pts_dst);
  cudaDeviceSynchronize();
  std::string Output_message ("GPU Voxelization Running Time: ");
  CLOCK_END(start,stop);
  
#if FILLING > 1 // Scaling
  float scale_factor=0.0f;

  CLOCK_START(start,stop);

  for(int a=0; a<SCALING_NUM; a++){
    scale_factor+=SCALING_STEPSIZE;
    _scaling_gpu_kernel<<<fullGrids, BLOCK_SIZE>>>(dst_device, dst_int_device, num_data_pts_dst, scale_factor);
  }
  cudaDeviceSynchronize();
  Output_message =("GPU Scaling Filling Running Time: ");
  CLOCK_END(start,stop);

#else  // Dilation
  bool *mark;
  check_return_status(cudaMalloc((void**)&mark, VOXEL_NUM * sizeof(bool)));

  CLOCK_START(start,stop);

  dim3 VoxelGrids(VOXEL_NUM_SINGLE ,VOXEL_NUM_SINGLE, VOXEL_NUM_SINGLE);
  dim3 th(6,1);

  for(int loop = 0; loop < LOOP_NUM; loop++)
    _dilation_gpu_kernel<<<VoxelGrids, th>>>(dst_int_device, mark, loop);
  cudaDeviceSynchronize();
  Output_message =("GPU Dilation Filling Running Time: ");
  CLOCK_END(start,stop);

#endif
}


__host__ void mem_int_initial( const float *src,  unsigned int * dst, int count){
 

  cudaEvent_t start, stop;
  CLOCK_START(start,stop);

  float x1=0, y1=0, z1=0;
  unsigned int x1_uint=0, y1_uint=0, z1_uint=0;
  for(int i = 0; i < count; i++){
    x1 = src[i];
    y1 = src[i+count];
    z1 = src[i+count*2];
    x1_uint = (unsigned int)(fma(x1,SCALE, SCALE));
    y1_uint = (unsigned int)(fma(y1,SCALE, SCALE));
    z1_uint = (unsigned int)(fma(z1,SCALE, SCALE));

    unsigned int index_block=0;
 
    index_block |= INDEX_VOXEL_X;
    index_block |= INDEX_VOXEL_Y;
    index_block |= INDEX_VOXEL_Z;

    int index_voxel = dst[index_block];
    dst[index_block+index_voxel*4+3]=i;
    dst[index_block+index_voxel*4+4]=x1_uint;
    dst[index_block+index_voxel*4+5]=y1_uint;
    dst[index_block+index_voxel*4+6]=z1_uint;
    dst[index_block]=index_voxel+1;

  }

  std::string Output_message ("CPU Voxelization Running Time: ");
  CLOCK_END(start,stop);

#if FILLING == 0  //Dilation_CPU
  bool *mark = (bool *)malloc(VOXEL_NUM*sizeof(bool));


  CLOCK_START(start,stop);
 
  for(int loop=0; loop<LOOP_NUM; loop++){
    for(int i = 0; i < VOXEL_NUM; i++){
      unsigned int index = (i<<(MEMORY_BITS-VOXEL_BITS_SINGLE*3));
      if(dst[index]==0){
	mark[i]=true;
      }else{
	mark[i]=false;
      }
    }
    for(int x = 0; x < VOXEL_NUM_SINGLE; x++){
      for(int y = 0; y < VOXEL_NUM_SINGLE; y++){
	for(int z = 0; z < VOXEL_NUM_SINGLE; z++){
	  unsigned int current_index = (x<<(VOXEL_BITS_SINGLE*2))|(y<<VOXEL_BITS_SINGLE)|z;
	  unsigned int index = (current_index << (MEMORY_BITS-VOXEL_BITS_SINGLE*3));
	  int i_temp;
	  unsigned int t;
	  if(dst[index]!=0&&mark[current_index]==false){
	    unsigned int root_index = loop==0? current_index:(dst[index]&VOXEL_MASK);
	    unsigned int updata_value=(0xff000000|root_index);
	    //-------x axis-----------
	    i_temp = (x-1)<0?0:(x-1);
	    t = (i_temp<<(MEMORY_BITS-VOXEL_BITS_SINGLE))|((current_index&CURRENT_INDEX_MASK_X)<<(MEMORY_BITS-VOXEL_BITS_SINGLE*3));
	    if(dst[t]==0)
	      dst[t]= updata_value;
	    i_temp = (x+1)>(VOXEL_NUM_SINGLE-1)?(VOXEL_NUM_SINGLE-1):(x+1);
	    t = (i_temp<<(MEMORY_BITS-VOXEL_BITS_SINGLE))|((current_index&CURRENT_INDEX_MASK_X)<<(MEMORY_BITS-VOXEL_BITS_SINGLE*3));
	    if(dst[t]==0)
	      dst[t]= updata_value;
 
	    //-------y axis-----------
	    i_temp = (y-1)<0?0:(y-1);
	    t = (i_temp<<(MEMORY_BITS-VOXEL_BITS_SINGLE*2))|((current_index&CURRENT_INDEX_MASK_Y)<<(MEMORY_BITS-VOXEL_BITS_SINGLE*3));
	    if(dst[t]==0)
	      dst[t]= updata_value;
	    i_temp = (y+1)>(VOXEL_NUM_SINGLE-1)?(VOXEL_NUM_SINGLE-1):(y+1);
	    t = (i_temp<<(MEMORY_BITS-VOXEL_BITS_SINGLE*2))|((current_index&CURRENT_INDEX_MASK_Y)<<(MEMORY_BITS-VOXEL_BITS_SINGLE*3));
	    if(dst[t]==0)
	      dst[t]= updata_value;
	    //-------z axis-----------
	    i_temp = (z-1)<0?0:(z-1);
	    t = (i_temp<<(MEMORY_BITS-VOXEL_BITS_SINGLE*3))|((current_index&CURRENT_INDEX_MASK_Z)<<(MEMORY_BITS-VOXEL_BITS_SINGLE*3));
	    if(dst[t]==0)
	      dst[t]= updata_value;
	    i_temp = (z+1)>(VOXEL_NUM_SINGLE-1)?(VOXEL_NUM_SINGLE-1):(z+1);
	    t = (i_temp<<(MEMORY_BITS-VOXEL_BITS_SINGLE*3))|((current_index&CURRENT_INDEX_MASK_Z)<<(MEMORY_BITS-VOXEL_BITS_SINGLE*3));
	    if(dst[t]==0)
	      dst[t]= updata_value;
 
	  }
#if VOXEL_TABLE_VISIABLE
	  printf("Dilating_num: %d, Voxel_index: %d; Voxel_add: %o; Root_Voxel_index: %d\n", loop, x*VOXEL_NUM_SINGLE*VOXEL_NUM_SINGLE+y*VOXEL_NUM_SINGLE+z, index, dst[index]&0x00ffffff);
#endif
	}
      }
    }



#if VOXEL_TABLE_VISIABLE
    // getchar();
#endif
  }
  Output_message =("CPU Dilation Filling Running Time: ");
  CLOCK_END(start,stop);
#elif FILLING == 2 // Scaling_CPU
  CLOCK_START(start,stop);

  unsigned int x1_scale, y1_scale, z1_scale;
  for(int i = 0; i < count; i++){
    x1 = src[i];
    y1 = src[i+count];
    z1 = src[i+count*2];
    x1_uint = (unsigned int)(fma(x1,SCALE, SCALE));
    y1_uint = (unsigned int)(fma(y1,SCALE, SCALE));
    z1_uint = (unsigned int)(fma(z1,SCALE, SCALE));

    float scale=0.0f;
    for(int loop = 1; loop < SCALING_NUM; loop++){
      scale +=SCALING_STEPSIZE;

      x1_scale = (unsigned int)(fma(x1,SCALE*scale, SCALE));
      y1_scale = (unsigned int)(fma(y1,SCALE*scale, SCALE));
      z1_scale = (unsigned int)(fma(z1,SCALE*scale, SCALE));
      
      if(x1_scale>2*SCALE||y1_scale>2*SCALE||z1_scale>2*SCALE)
      	continue;
      else if(x1_scale<0||y1_scale<0||z1_scale<0)
	continue;
      unsigned int index_block=0;
      index_block |= INDEX_VOXEL_SCALE_X ;
      index_block |= INDEX_VOXEL_SCALE_Y ;
      index_block |= INDEX_VOXEL_SCALE_Z ;

      unsigned int index_voxel = dst[index_block];
      unsigned int index_voxel_filling = dst[index_block+1];
      if(index_voxel==0){	
	if(index_voxel_filling==0||dst[index_block+(index_voxel_filling-1)*4+3]!=i){	 
	  dst[index_block+index_voxel_filling*4+3]=i;
	  dst[index_block+index_voxel_filling*4+4]=x1_uint;
	  dst[index_block+index_voxel_filling*4+5]=y1_uint;
	  dst[index_block+index_voxel_filling*4+6]=z1_uint;
	  dst[index_block+1]=index_voxel_filling+1;
	}
      }
      //printf("Scaling_num: %d, Voxel_index: %d; Root_pix_index: %d, count: %d\n", loop, index_block>>14,i,dst[index_block+1]);

    }
    //getchar();
  }
  Output_message =("CPU Scaling Filling Running Time: ");
  CLOCK_END(start,stop);

#endif
}



/***************************       Main  algorithm *********************************/
__host__ int icp_cuda(const Eigen::MatrixXf &dst,  const Eigen::MatrixXf &src, int max_iterations, float tolerance, Eigen::MatrixXf &src_transformed, NEIGHBOR &neighbor_out){
  assert(src_transformed.cols() == 4 && src_transformed.rows() == src.rows());
  //assert(src.rows() == dst.rows());// && dst.rows() == dst_chorder.rows());
  assert(src.cols() == dst.cols());// && dst.cols() == dst_chorder.cols());
  assert(dst.cols() == 3);
  //assert(dst.rows() == neighbor.indices.size());

  std::deque<Eigen::MatrixXd> residual_history;


  // Host variables declaration
  int num_data_pts = src.rows();
  int num_data_pts_dst = dst.rows();
  const float *dst_host         = dst.data();
  const float *src_host         = src.data();
  float *gpu_temp_res          = src_transformed.data();
  int *best_neighbor_host = (int *)malloc(num_data_pts*sizeof(int));
  double *best_dist_host  = (double *)malloc(num_data_pts*sizeof(double));


  // Device variables declaration
  float *dst_chorder_device, *dst_device, *src_device, *src_4d_device;
  float *src_4d_t_device; // temp result
  float *dst_chorder_zm_device, *src_zm_device;
  int *neighbor_device;

  //int *best_neighbor_device;
  double *best_dist_device;

  // CUBLAS and CUSOLVER initialization
  // Create a handle for CUBLAS
  cublasHandle_t handle;
  cublasCreate(&handle);

  // CUDA solver initialization
  cusolverDnHandle_t solver_handle;
  cusolverDnCreate(&solver_handle);

  float *ones_host = (float *)malloc(num_data_pts*sizeof(float));
  for(int i = 0; i< num_data_pts; i++){
    ones_host[i] = 1;}
  float *average_host = (float *)malloc(3*sizeof(float));
  float *ones_device, *sum_device_src, *sum_device_dst;


  unsigned int *mem_int_host=(unsigned int *)malloc(MEMORY_SIZE*sizeof(unsigned int));
  unsigned int *mem_int_device;
  unsigned int *mem_int_comp_device;
  unsigned int *mem_addstart_comp_device;
  unsigned int *mem_hist_device;
#if NN_OPTIMIZE == 2

#if FILLING == 1 || FILLING == 3
  check_return_status(cudaMallocManaged((void**)&mem_hist_device, VOXEL_NUM*sizeof(unsigned int)));
  check_return_status(cudaMalloc((void**)&mem_int_device, MEMORY_SIZE * sizeof(unsigned int)));
  check_return_status(cudaMalloc((void**)&mem_addstart_comp_device, VOXEL_NUM*sizeof(unsigned int)));
  cudaMemset(mem_hist_device, 0, VOXEL_NUM*sizeof(unsigned int)); 

#elif FILLING == 0 || FILLING==2
  cudaEvent_t start, stop;
  CLOCK_START(start,stop);
  mem_int_initial(dst_host, mem_int_host, num_data_pts_dst);
  check_return_status(cudaMalloc((void**)&mem_int_device, MEMORY_SIZE * sizeof(unsigned int)));
  check_return_status(cudaMemcpy(mem_int_device, mem_int_host, MEMORY_SIZE * sizeof(unsigned int), cudaMemcpyHostToDevice));

#if FILLING ==0
  std::string Output_message ("Dilation CPU Time: ");
#elif FILLING ==2
  std::string Output_message ("Scaling CPU Time: ");
#endif

  CLOCK_END(start,stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop); 
#endif
#endif
   
  /*************************       CUDA memory operations          ********************************/
  // Initialize the CUDA memory
  check_return_status(cudaMalloc((void**)&dst_device         , 3 * num_data_pts_dst * sizeof(float)));
  check_return_status(cudaMalloc((void**)&dst_chorder_device , 3 * num_data_pts * sizeof(float)));
  check_return_status(cudaMalloc((void**)&src_device         , 3 * num_data_pts * sizeof(float)));
  check_return_status(cudaMalloc((void**)&src_4d_device      , 4 * num_data_pts * sizeof(float)));
  check_return_status(cudaMalloc((void**)&src_4d_t_device, 4 * num_data_pts * sizeof(float)));

  check_return_status(cudaMalloc((void**)&neighbor_device   , num_data_pts * sizeof(int)));
  check_return_status(cudaMalloc((void**)&dst_chorder_zm_device, 3 * num_data_pts * sizeof(float)));
  check_return_status(cudaMalloc((void**)&src_zm_device, 3 * num_data_pts * sizeof(float)));

  check_return_status(cudaMalloc((void**)&ones_device, num_data_pts * sizeof(float)));
  check_return_status(cudaMalloc((void**)&sum_device_src, 3 * sizeof(float)));
  check_return_status(cudaMalloc((void**)&sum_device_dst, 3 * sizeof(float)));
   
  //check_return_status(cudaMalloc((void**)&best_neighbor_device, num_data_pts * sizeof(int)));
  check_return_status(cudaMalloc((void**)&best_dist_device, num_data_pts * sizeof(double)));


  // Copy data from host to device
  check_return_status(cudaMemcpy(dst_device, dst_host, 3 * num_data_pts_dst * sizeof(float), cudaMemcpyHostToDevice));
  check_return_status(cudaMemcpy(src_device, src_host, 3 * num_data_pts * sizeof(float), cudaMemcpyHostToDevice));
  //check_return_status(cudaMemcpy(neighbor_device, &(neighbor.indices[0]),  num_data_pts * sizeof(int), cudaMemcpyHostToDevice));
   
  check_return_status(cublasSetVector(num_data_pts, sizeof(float), ones_host, 1, ones_device, 1));
  check_return_status(cudaMemcpy(src_4d_device, src_device, 3 * num_data_pts * sizeof(float), cudaMemcpyDeviceToDevice));
  check_return_status(cudaMemcpy(src_4d_device + 3 * num_data_pts,
				 ones_device, num_data_pts * sizeof(float), cudaMemcpyDeviceToDevice));
   
  /*******************************    Actual work done here                   ********************************/
  double prev_error = 0;
  double mean_error = 0;
  unsigned mem_size=0;
#if NN_OPTIMIZE==2 && (FILLING == 1||FILLING==3)  
  cudaEvent_t start_gpu, stop_gpu;
  CLOCK_START(start_gpu,stop_gpu);
  mem_size=_hist_asyc(dst_device, mem_hist_device, num_data_pts_dst)+500;
  printf("%d\n", mem_size);
  check_return_status(cudaMemcpy(mem_addstart_comp_device, mem_hist_device, VOXEL_NUM * sizeof(unsigned int), cudaMemcpyHostToDevice));
  check_return_status(cudaMalloc((void**)&mem_int_comp_device, mem_size * sizeof(unsigned int)));
  cudaMemset(mem_int_comp_device, 0, mem_size*sizeof(unsigned int));

  //_filling_gpu(dst_device, mem_int_device, num_data_pts_dst);
  _filling_comp_gpu(dst_device, mem_int_comp_device, mem_addstart_comp_device, num_data_pts_dst);
 
  
#if FILLING == 1
  std::string Output_message ("Dilation GPU Time: ");
#elif FILLING ==3
  std::string Output_message ("Scaling GPU Time: ");
#endif
  CLOCK_END(start_gpu,stop_gpu);
  cudaEventDestroy(start_gpu);
  cudaEventDestroy(stop_gpu);
#endif

  _nearest_neighbor_cuda_warper(src_4d_device, dst_device, mem_addstart_comp_device,  mem_int_comp_device, num_data_pts, num_data_pts_dst, best_dist_device, neighbor_device, 0);

  check_return_status(cublasDasum(handle, num_data_pts, best_dist_device, 1, &prev_error));
  prev_error /= num_data_pts;

  //float tolerance = 1e-6;
  int iter = 0;
  for(int i = 0; i <max_iterations; i++){
   
    //sleep(1);
    _apply_optimal_transform_cuda_warper(handle, solver_handle, dst_device, src_device, neighbor_device, ones_device, num_data_pts, num_data_pts_dst, //const input
					 dst_chorder_device, dst_chorder_zm_device, src_zm_device, sum_device_dst, sum_device_src, // temp cache only
					 src_4d_t_device, src_4d_device, residual_history // results we care
					 );
    cudaEvent_t start, stop;
    CLOCK_START(start,stop);
    
    std::string Output_message ("Running Time: ");
    CLOCK_END(start,stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    //src_4d_device stored in col major, shape is (num_pts, 3)
    _nearest_neighbor_cuda_warper(src_4d_device, dst_device, mem_addstart_comp_device, mem_int_comp_device, num_data_pts, num_data_pts_dst, best_dist_device, neighbor_device, i);
       
       
    check_return_status(cudaMemcpy(src_device, src_4d_device, 3* num_data_pts * sizeof(float), cudaMemcpyDeviceToDevice));
       
    check_return_status(cublasDasum(handle, num_data_pts, best_dist_device, 1, &mean_error));
    mean_error /= num_data_pts;
// #if NN_OPTIMIZE == 2
//     mean_error /= SCALE;
// #endif
    std::cout << "mean_error:"<< mean_error  << std::endl;
    if ((abs(prev_error - mean_error) < tolerance)||(i==max_iterations-1)){
       std::cout << "mean_error: "<< mean_error  << std::endl;
       break;
    }
    // Calculate mean error and compare with previous error
    prev_error = mean_error;    
    iter = i + 2;
  }
   
  check_return_status(cudaMemcpy(best_neighbor_host, neighbor_device, num_data_pts * sizeof(int), cudaMemcpyDeviceToHost));
  check_return_status(cudaMemcpy(best_dist_host    , best_dist_device    , num_data_pts * sizeof(double), cudaMemcpyDeviceToHost));

  neighbor_out.distances.clear();
  neighbor_out.indices.clear();
  for(int i = 0; i < num_data_pts; i++){
    neighbor_out.distances.push_back(best_dist_host[i]);
    neighbor_out.indices.push_back(best_neighbor_host[i]);
  }


   
  /**********************************  Final cleanup steps     ********************************************/
  // Destroy the handle
  cublasDestroy(handle);
  cusolverDnDestroy(solver_handle);

  // Final result copy back
  check_return_status(cudaMemcpy(gpu_temp_res, src_4d_device, 4 * num_data_pts * sizeof(float), cudaMemcpyDeviceToHost));
  // check_return_status(cudaMemcpy(gpu_temp_res, trans_matrix_device, 4 * 4 * sizeof(double), cudaMemcpyDeviceToHost));
  

  // Free all variables
  std::cout<<"HERE"<<std::endl;
  check_return_status(cudaFree(dst_device));
  check_return_status(cudaFree(src_device));
  check_return_status(cudaFree(dst_chorder_device));
  check_return_status(cudaFree(neighbor_device));
  check_return_status(cudaFree(dst_chorder_zm_device));
  check_return_status(cudaFree(src_zm_device));
  check_return_status(cudaFree(ones_device));
#if NN_OPTIMIZE == 2
  check_return_status(cudaFree(mem_int_device));
  check_return_status(cudaFree(mem_hist_device));
  check_return_status(cudaFree(mem_addstart_comp_device));
  check_return_status(cudaFree(mem_int_comp_device));
  
#endif
  return iter;
}


