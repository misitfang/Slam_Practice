#ifndef ICP3D3D_FACTOR_H_
#define ICP3D3D_FACTOR_H_
#include <Eigen/Core>

using namespace Eigen;
class Icp3d3d_factor
{

public:
    static void icp3d3d_jacobi(const Vector3d &points1,
                               const Vector3d &points2,
                               Matrix3d &R,
                               Vector3d &t,
                               Matrix<double, 3, 6> &J);
    static void icp3d3d_error(const Vector3d &points1,
                              const Vector3d &points2,
                              Matrix3d &R,
                              Vector3d &t,
                              Vector3d &error);
};
#endif
