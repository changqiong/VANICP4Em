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

#ifndef DEF_H
#define DEF_H

//*******************Adjustable Values*******************//
#define VOXEL_BITS_SINGLE    4 
#define LOOP_NUM 20 

//*******************Fixed Values*******************//
#if VOXEL_BITS_SINGLE==4

#define SCALE 65536.0f
#define INDEX_VOXEL_HIST_X ((x1_uint&0x1E000)>>5)   
#define INDEX_VOXEL_HIST_Y ((y1_uint&0x1E000)>>9)   
#define INDEX_VOXEL_HIST_Z ((z1_uint&0x1E000)>>13)  


#define VOXEL_MASK 0xFFF
#define VOXEL_NUM_SINGLE 16  //voxel number in single axis
#define VOXEL_NUM 4096  //total voxel number
#define CURRENT_INDEX_MASK_X 0xFF
#define CURRENT_INDEX_MASK_Y 0xF0F
#define CURRENT_INDEX_MASK_Z 0xFF0
#endif


#if VOXEL_BITS_SINGLE==3

#define SCALE 65536.0f
#define INDEX_VOXEL_HIST_X ((x1_uint&0x1C000)>>8)   
#define INDEX_VOXEL_HIST_Y ((y1_uint&0x1C000)>>11)   
#define INDEX_VOXEL_HIST_Z ((z1_uint&0x1C000)>>14)  


#define VOXEL_MASK 0x1FF
#define VOXEL_NUM_SINGLE 8  //voxel number in single axis
#define VOXEL_NUM 512  //total voxel number
#define CURRENT_INDEX_MASK_X 0x3F
#define CURRENT_INDEX_MASK_Y 0x1C7
#define CURRENT_INDEX_MASK_Z 0x1F8
#endif


#endif
