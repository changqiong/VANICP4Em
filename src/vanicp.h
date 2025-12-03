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

#include "Eigen/Eigen"
#include <vector>
#include "def.h"


#ifndef VANICP_H
#define VANICP_H

#define INF 1e40

typedef struct{
    Eigen::Matrix4d trans;
    std::vector<float> distances;
    int iter;
}  ICP_OUT;

typedef struct{
    std::vector<float> distances;
    std::vector<int> indices;
} NEIGHBOR;

int icp_cuda(
	     const Eigen::MatrixXf& src,
	     const Eigen::MatrixXf& dst,
	     Eigen::MatrixXf&       src_transformed,
	     NEIGHBOR&           neighbor_out,
	     int                    max_iterations,
	     double                 tolerance);

#endif
