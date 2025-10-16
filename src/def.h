#ifndef DEF_H
#define DEF_H

//#include "temp.h"
//*******************Adjustable Values*******************//
#define NN_OPTIMIZE 2
#define FILLING 1 // Range [0,3] 0: Dilation_CPU; 1: Dilation_GPU; 2: Scaling_CPU; 3: Scaling_GPU;  
#define SCALING_NUM 200
#define SCALING_STEPSIZE 0.01f
#define VOXEL_BITS_SINGLE    4  //Range: 1~8. Voxel bits number in single axis. 
#define LOOP_NUM 10  // Range: [0~).  
#define MATRIX_VISIABLE false
#define VOXEL_TABLE_VISIABLE false 

//*******************Fixed Values*******************//
#define INT_COOR_BITS 17 //Fixed Value
#define MEMORY_BITS 26  // Fixed value. Memory address bits number 
#define MEMORY_SIZE 67108864 //Fixed Value

#if VOXEL_BITS_SINGLE==8

#if INT_COOR_BITS==17
#define SCALE 65536.0f
#define INT_COOR_MASK 0x1FE00
#define INDEX_VOXEL_X ((x1_uint&0x1FE00)<<9)   
#define INDEX_VOXEL_Y ((y1_uint&0x1FE00)<<1)   
#define INDEX_VOXEL_Z ((z1_uint&0x1FE00)>>7)  
#define INDEX_VOXEL_SCALE_X ((x1_scale&0x1FE00)<<9)   
#define INDEX_VOXEL_SCALE_Y ((y1_scale&0x1FE00)<<1)   
#define INDEX_VOXEL_SCALE_Z ((z1_scale&0x1FE00)>>7)  
#endif

#define VOXEL_MASK 0xFFFFFF
#define VOXEL_NUM_SINGLE 256  //voxel number in single axis
#define VOXEL_NUM 16777216  //total voxel number
#define CURRENT_INDEX_MASK_X 0xFFFF
#define CURRENT_INDEX_MASK_Y 0xFF00FF
#define CURRENT_INDEX_MASK_Z 0xFFFF00
#endif



#if VOXEL_BITS_SINGLE==7

#if INT_COOR_BITS==17
#define SCALE 65536.0f
#define INT_COOR_MASK 0x1FC00
#define INDEX_VOXEL_X ((x1_uint&0x1FC00)<<9)   
#define INDEX_VOXEL_Y ((y1_uint&0x1FC00)<<2)   
#define INDEX_VOXEL_Z ((z1_uint&0x1FC00)>>5)   
#define INDEX_VOXEL_SCALE_X ((x1_scale&0x1FC00)<<9)   
#define INDEX_VOXEL_SCALE_Y ((y1_scale&0x1FC00)<<2)   
#define INDEX_VOXEL_SCALE_Z ((z1_scale&0x1FC00)>>5) 
#endif


#define VOXEL_MASK 0x1FFFFF
#define VOXEL_NUM_SINGLE 128  //voxel number in single axis
#define VOXEL_NUM 2097152  //total voxel number
#define CURRENT_INDEX_MASK_X 0x3FFF
#define CURRENT_INDEX_MASK_Y 0x1FC07F
#define CURRENT_INDEX_MASK_Z 0x1FFF80
#endif



#if VOXEL_BITS_SINGLE==6

#if INT_COOR_BITS==17
#define SCALE 65536.0f
#define INT_COOR_MASK 0x1F800
#define INDEX_VOXEL_X ((x1_uint&0x1F800)<<9)   
#define INDEX_VOXEL_Y ((y1_uint&0x1F800)<<3)   
#define INDEX_VOXEL_Z ((z1_uint&0x1F800>>3))   
#define INDEX_VOXEL_SCALE_X ((x1_scale&0x1F800)<<9)   
#define INDEX_VOXEL_SCALE_Y ((y1_scale&0x1F800)<<3)   
#define INDEX_VOXEL_SCALE_Z ((z1_scale&0x1F800)>>3) 
#endif


#define VOXEL_MASK 0x3FFFF
#define VOXEL_NUM_SINGLE 64  //voxel number in single axis
#define VOXEL_NUM 262144  //total voxel number
#define CURRENT_INDEX_MASK_X 0xFFF
#define CURRENT_INDEX_MASK_Y 0X3F03F
#define CURRENT_INDEX_MASK_Z 0x3FFC0
#endif


#if VOXEL_BITS_SINGLE==5


#if INT_COOR_BITS==17
#define SCALE 65536.0f
#define INT_COOR_MASK 0x1F000
#define INDEX_VOXEL_X ((x1_uint&0x1F000)<<9)   
#define INDEX_VOXEL_Y ((y1_uint&0x1F000)<<4)   
#define INDEX_VOXEL_Z ((z1_uint&0x1F000)>>1)   
#define INDEX_VOXEL_HIST_X ((x1_uint&0x1F000)>>2)   
#define INDEX_VOXEL_HIST_Y ((y1_uint&0x1F000)>>7)   
#define INDEX_VOXEL_HIST_Z ((z1_uint&0x1F000)>>12)   
#define INDEX_VOXEL_SCALE_X ((x1_scale&0x1F000)<<9)   
#define INDEX_VOXEL_SCALE_Y ((y1_scale&0x1F000)<<4)   
#define INDEX_VOXEL_SCALE_Z ((z1_scale&0x1F000)>>1)   
#endif


#define VOXEL_MASK 0x7FFF
#define VOXEL_NUM_SINGLE 32  //voxel number in single axis
#define VOXEL_NUM 32768  //total voxel number
#define CURRENT_INDEX_MASK_X 0x3FF
#define CURRENT_INDEX_MASK_Y 0X7C1F
#define CURRENT_INDEX_MASK_Z 0x7FE0
#endif

#if VOXEL_BITS_SINGLE==4

#if INT_COOR_BITS==17
#define SCALE 65536.0f
#define INT_COOR_MASK 0x1E000
#define INDEX_VOXEL_X ((x1_uint&0x1E000)<<9)   
#define INDEX_VOXEL_Y ((y1_uint&0x1E000)<<5)   
#define INDEX_VOXEL_Z ((z1_uint&0x1E000)<<1)   
#define INDEX_VOXEL_HIST_X ((x1_uint&0x1E000)>>5)   
#define INDEX_VOXEL_HIST_Y ((y1_uint&0x1E000)>>9)   
#define INDEX_VOXEL_HIST_Z ((z1_uint&0x1E000)>>13)  
#define INDEX_VOXEL_SCALE_X ((x1_scale&0x1E000)<<9)   
#define INDEX_VOXEL_SCALE_Y ((y1_scale&0x1E000)<<5)   
#define INDEX_VOXEL_SCALE_Z ((z1_scale&0x1E000)<<1)   
#endif


#define VOXEL_MASK 0xFFF
#define VOXEL_NUM_SINGLE 16  //voxel number in single axis
#define VOXEL_NUM 4096  //total voxel number
#define CURRENT_INDEX_MASK_X 0xFF
#define CURRENT_INDEX_MASK_Y 0XF0F
#define CURRENT_INDEX_MASK_Z 0xFF0
#endif



#if VOXEL_BITS_SINGLE==3

#if INT_COOR_BITS==17
#define SCALE 65536.0f
#define INT_COOR_MASK 0x1C000
#define INDEX_VOXEL_X ((x1_uint&0x1C000)<<9)   
#define INDEX_VOXEL_Y ((y1_uint&0x1C000)<<6)   
#define INDEX_VOXEL_Z ((z1_uint&0x1C000)<<3)   
#define INDEX_VOXEL_SCALE_X ((x1_scale&0x1C000)<<9)   
#define INDEX_VOXEL_SCALE_Y ((y1_scale&0x1C000)<<6)   
#define INDEX_VOXEL_SCALE_Z ((z1_scale&0x1C000)<<3)   
#endif


#define VOXEL_MASK 0x1FF
#define VOXEL_NUM_SINGLE 8  //voxel number in single axis
#define VOXEL_NUM 512  //total voxel number
#define CURRENT_INDEX_MASK_X 0x3F
#define CURRENT_INDEX_MASK_Y 0X1C7
#define CURRENT_INDEX_MASK_Z 0x1F8
#endif


#if VOXEL_BITS_SINGLE==2

#if INT_COOR_BITS==17
#define SCALE 65536.0f
#define INT_COOR_MASK 0x18000
#define INDEX_VOXEL_X ((x1_uint&0x18000)<<9)   
#define INDEX_VOXEL_Y ((y1_uint&0x18000)<<7)   
#define INDEX_VOXEL_Z ((z1_uint&0x18000)<<5)   
#define INDEX_VOXEL_SCALE_X ((x1_scale&0x18000)<<9)   
#define INDEX_VOXEL_SCALE_Y ((y1_scale&0x18000)<<7)   
#define INDEX_VOXEL_SCALE_Z ((z1_scale&0x18000)<<5)   
#endif


#define VOXEL_MASK 0x3F
#define VOXEL_NUM_SINGLE 4  //voxel number in single axis
#define VOXEL_NUM 64  //total voxel number
#define CURRENT_INDEX_MASK_X 0xF
#define CURRENT_INDEX_MASK_Y 0X33
#define CURRENT_INDEX_MASK_Z 0x3C
#endif

#if VOXEL_BITS_SINGLE==1

#if INT_COOR_BITS==17
#define SCALE 65536.0f
#define INT_COOR_MASK 0x10000
#define INDEX_VOXEL_X ((x1_uint&0x10000)<<9)   
#define INDEX_VOXEL_Y ((y1_uint&0x10000)<<8)   
#define INDEX_VOXEL_Z ((z1_uint&0x10000)<<7)   
#define INDEX_VOXEL_SCALE_X ((x1_scale&0x10000)<<9)   
#define INDEX_VOXEL_SCALE_Y ((y1_scale&0x10000)<<8)   
#define INDEX_VOXEL_SCALE_Z ((z1_scale&0x10000)<<7)   
#endif


#define VOXEL_MASK 0x3F
#define VOXEL_NUM_SINGLE 2  //voxel number in single axis
#define VOXEL_NUM 8  //total voxel number
#define CURRENT_INDEX_MASK_X 0x3
#define CURRENT_INDEX_MASK_Y 0X5
#define CURRENT_INDEX_MASK_Z 0x6
#endif


#endif
