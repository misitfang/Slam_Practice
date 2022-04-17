#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include "icp3d3d_factor.h"
#include "utility.h"

using namespace Eigen;

void Icp3d3d_factor::icp3d3d_error(
    const Vector3d &point1,
    const Vector3d &point2, Matrix<double, 3, 3> &R,
    Vector3d &t,
    Matrix<double, 3, 1> &error)
{
    error = point1 - R * point2 - t;
}
void Icp3d3d_factor::icp3d3d_jacobi(const Vector3d &point1,
                                    const Vector3d &point2,
                                    Matrix3d &R,
                                    Vector3d &t,
                                    Matrix<double, 3, 6> &J)
{   
    Vector3d A = R *point2;
    Matrix3d J1_dR;
    Utility::Vector3d_Skew_Symmetric_Matrix(A,J1_dR);
    Matrix3d J2_dt = -Matrix3d::Identity();
    J<<J1_dR,J2_dt;
}