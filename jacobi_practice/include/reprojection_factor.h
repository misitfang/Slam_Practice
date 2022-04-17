#ifndef REPROJECTION_FACTOR_H_
#define REPROJECTION_FACTOR_H_
#include <Eigen/Core>
#include <opencv2/core/core.hpp>
#include "sophus/se3.hpp"


using namespace std;
using namespace cv;
using namespace Eigen;

class ReprojectionFactor
{
public:
    static void reprojection_jacobi(const Vector3d &points_3d,
                                    const Vector2d &points_2d, const Mat &K,
                                    Sophus::SE3d &pose,
                                    Matrix<double, 2, 6> &J_3d2d);
    static void reprojection_error(const Vector3d &points_3d,
                                   const Vector2d &points_2d,
                                   const Mat &K,
                                   Sophus::SE3d &pose,
                                   Vector2d &error3d2d);
};
#endif
