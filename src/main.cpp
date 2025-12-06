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
#include <vector>
#include <numeric>
#include "Eigen/Eigen"
#include <random>
#include <sys/time.h>
#include <chrono>
#include "vanicp.h"
#include "io.h"

using namespace std;
using namespace Eigen;


Matrix4d transform_SVD(const MatrixXd& A,
                              const MatrixXd& B)
{

    const int n = static_cast<int>(A.rows());
    assert(B.rows() == n && "A and B must have the same number of points");
    assert(A.cols() == 3 && B.cols() == 3 && "A and B must be N×3 point sets");

    // Compute centroids
    const Vector3d centroid_A = A.colwise().mean();
    const Vector3d centroid_B = B.colwise().mean();

    // Subtract centroids (zero-mean point sets)
    MatrixXd A_zm = A.rowwise() - centroid_A.transpose();
    MatrixXd B_zm = B.rowwise() - centroid_B.transpose();

    // 3×3 cross-covariance matrix
    const Matrix3d H = A_zm.transpose() * B_zm;

    // SVD of H
    JacobiSVD<Matrix3d> svd(H, ComputeFullU | ComputeFullV);
    Matrix3d U = svd.matrixU();
    Matrix3d V = svd.matrixV();

    // Rotation
    Matrix3d R = V * U.transpose();

    // Handle reflection
    if (R.determinant() < 0.0) {
        V.col(2) *= -1.0;
        R = V * U.transpose();
    }

    // Translation
    const Vector3d t = centroid_B - R * centroid_A;

    // Assemble 4×4 transform
    Matrix4d T = Matrix4d::Identity();
    T.topLeftCorner<3,3>() = R;
    T.topRightCorner<3,1>() = t;

    return T;
}

ICP_OUT icp(const MatrixXd &A,
            const MatrixXd &B,
            int max_iterations,
            float tolerance)
{
    // A, B: [N x 3] double, each row is a 3D point
    const int num_points = static_cast<int>(A.rows());

    // Convert to float and transpose → [3 x N]
    MatrixXf src3f = A.cast<float>().transpose();  // source
    MatrixXf dst3f = B.cast<float>().transpose();  // target

    // Homogeneous source points: [4 x N]
    MatrixXf src_h = MatrixXf::Ones(4, num_points);
    src_h.topRows<3>() = src3f;

    // Buffer for CUDA ICP output: [N x 4]
    MatrixXf src_next(num_points, 4);

    NEIGHBOR neighbor;
    ICP_OUT result;

    // Run CUDA-ICP (input: [N x 3], output: [N x 4])
    int iter = vanicp(
                        src3f.transpose(),    // [N x 3]
                        dst3f.transpose(),    // [N x 3]
			src_next,
                        neighbor,
			max_iterations,
                        tolerance
                        );

    // Update homogeneous source with aligned points
    src_h = src_next.transpose();             // [4 x N]

    // Take only XYZ, convert back to double, shape [N x 3]
    MatrixXd src_aligned =
        src_h.topRows<3>().cast<double>().transpose();

    // Compute final rigid transform using SVD
    Matrix4d T = transform_SVD(A, src_aligned);

    // Pack result
    result.trans      = T;
    result.distances  = neighbor.distances;
    result.iter       = iter;

    return result;
}





int main(int argc, char* argv[]) {
    // --- Argument check ---
    if (argc != 4) {
        cerr << "Usage: " << argv[0]
                  << " path_to_base_dir original_file_path translated_file_path\n";
        return EXIT_FAILURE;
    }

    string base_dir = argv[1];
    // Ensure base_dir ends with a slash if needed
    if (!base_dir.empty() && base_dir.back() != '/' && base_dir.back() != '\\') {
        base_dir += "/";
    }

    const string original_path   = base_dir + argv[2];
    const string translated_path = base_dir + argv[3];

    // --- Load point clouds ---
    MatrixXd A = load_pcl(original_path);
    MatrixXd B = load_pcl(translated_path);

    // --- Run ICP ---
    double total_time_sec = 0.0;

    const auto t_start = chrono::steady_clock::now();
    ICP_OUT icp_result = icp(B, A, /*max_iter=*/50, /*tol=*/1e-6);
    const auto t_end   = chrono::steady_clock::now();

    total_time_sec =
        chrono::duration_cast<chrono::duration<double>>(t_end - t_start).count();

    Matrix4d   T   = icp_result.trans;
    const auto&     dist = icp_result.distances;
    const int       iter = icp_result.iter;
    Vector3d t = T.block<3,1>(0,3);
    Matrix3d R = T.block<3,3>(0,0);

    cout << "Iteration #: " << iter << "\n";
    cout << "ICP runtime: " << total_time_sec << " s\n";
    if (iter > 0) {
        cout << "Each iteration takes " << (total_time_sec / iter) << " s\n";
    }
   
    // --- Reconstruct the point cloud ---
    MatrixXd D = (R * B.transpose()).transpose();
    D.rowwise() += t.transpose();

    cout << "Transformed point cloud saved.\n";
    save_pcl(base_dir + "transformed.txt", D);

    cout << "Transformation saved.\n";
    save_tranformation(base_dir + "transformation.txt", T);

    return EXIT_SUCCESS;
}

