#include <iostream>
#include <Eigen/Core>
#include "reprojection_factor.h"
#include "sophus/se3.hpp"
#include <opencv2/core/core.hpp>

using namespace Eigen;
using namespace std;
using namespace cv;

void ReprojectionFactor::reprojection_error(const Vector3d &points_3d,
                                            const Vector2d &points_2d,
                                            const Mat &k,
                                            Sophus::SE3d &pose,
                                            Vector2d &error3d2d)
{
     double fx = k.at<double>(0, 0);
     double fy = k.at<double>(1, 1);
     double cx = k.at<double>(0, 2);
     double cy = k.at<double>(1, 2);
     Vector3d p = pose * points_3d;
     Vector2d projection(fx * p[0] / p[2] + cx, fy * p[1] / p[2] + cy);
     error3d2d = projection - points_2d;
}

void ReprojectionFactor::reprojection_jacobi(const Vector3d &points_3d, const Vector2d &points_2d, const Mat &k,
                                             Sophus::SE3d &pose, Matrix<double, 2, 6> &J_3d2d)
{
     double fx = k.at<double>(0, 0);
     double fy = k.at<double>(1, 1);
     double cx = k.at<double>(0, 2);
     double cy = k.at<double>(1, 2);
     Vector3d p = pose * points_3d;
     double pzsquare = p[2] * p[2];
     J_3d2d << fx / p[2], 0, -fx * p[0] / pzsquare, -fx * p[0] * p[1] / pzsquare, fx * (p[2] * p[2] + p[0] * p[0]) / pzsquare, -fx * p[1] / p[2],
         0, fy / p[2], -fy * p[1] / pzsquare, -fy * (p[2] * p[2] + p[1] * p[1]) / pzsquare, fy * p[0] * p[1] / pzsquare, fy * p[0] / pzsquare;
}
