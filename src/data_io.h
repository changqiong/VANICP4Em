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

#define MAXBUFSIZE  ((int) 1e6)

using namespace Eigen;

MatrixXd load_pcl(std::string file_name, int col = 3);
void save_pcl(std::string file_name, MatrixXd& pcl_data);
void save_tranformation(std::string file_name, Matrix4d& transformation);
