#include <iostream>
#include "reprojection_factor.h"
#include "utility.h"
#include "GetPoints.h"
#include <Eigen/Core>
#include "sophus/se3.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>


using namespace cv;
using namespace std;
using namespace Eigen;  

typedef vector<Vector3d, Eigen::aligned_allocator<Vector3d>> Vecvector3d;
typedef vector<Vector2d, Eigen::aligned_allocator<Vector2d>> Vecvector2d;

Point2d pixel2cam(const Point2d &p, const Mat &K)
{
    return Point2d(
        (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
        (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1));
}
int main(int argc, char **argv)
{
    if (argc != 5)
    {
        cout << "usage: pose_estimation_3d2d img1 img2 depth1 depth2" << endl;
        return 1;
    }
    //-- 读取图像
    Mat img_1 = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    Mat img_2 = imread(argv[2], CV_LOAD_IMAGE_COLOR);
    assert(img_1.data && img_2.data && "Can not load images!");

    vector<KeyPoint> keypoints_1, keypoints_2;
    vector<DMatch> matches;
    GetPoints::find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches);
    cout << "一共找到了" << matches.size() << "组匹配点" << endl;

    // 建立3D点
    Mat d1 = imread(argv[3], CV_LOAD_IMAGE_UNCHANGED); // 深度图为16位无符号数，单通道图像
    Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    vector<Point3f> pts_3d;
    vector<Point2f> pts_2d;
    for (DMatch m : matches)
    {
        ushort d = d1.ptr<unsigned short>(int(keypoints_1[m.queryIdx].pt.y))[int(keypoints_1[m.queryIdx].pt.x)];
        if (d == 0) // bad depth
            continue;
        float dd = d / 5000.0;
        Point2d p1 = pixel2cam(keypoints_1[m.queryIdx].pt, K);
        pts_3d.push_back(Point3f(p1.x * dd, p1.y * dd, dd));
        pts_2d.push_back(keypoints_2[m.trainIdx].pt);
    }
    // cout << pts_3d << endl;
    // cout << pts_2d << endl;
    Vecvector3d pts_3d_eigen;
    Vecvector2d pts_2d_eigen;
    for (size_t i = 0; i < pts_3d.size(); ++i)
    {
        pts_3d_eigen.push_back(Vector3d(pts_3d[i].x, pts_3d[i].y, pts_3d[i].z));
        pts_2d_eigen.push_back(Vector2d(pts_2d[i].x, pts_2d[i].y));
    }
    Sophus::SE3d pose_3d2d;
    cout << "2\n"
         << pose_3d2d.matrix() << endl;
    Vector2d error_;
    Matrix<double, 2, 6> jacobi3d2d;
    const int iterations = 10;
    double cost = 0, lastcost = 0;
    for (int iter = 0; iter < iterations; iter++)
    {
        Matrix<double, 6, 6> H = Matrix<double, 6, 6>::Zero();
        Matrix<double, 6, 1> b = Matrix<double, 6, 1>::Zero();

        cost = 0;
        for (int i = 0; i < pts_3d_eigen.size(); i++)
        {
            ReprojectionFactor::reprojection_error(pts_3d_eigen[i], pts_2d_eigen[i], K, pose_3d2d, error_);
            ReprojectionFactor::reprojection_jacobi(pts_3d_eigen[i], pts_2d_eigen[i], K, pose_3d2d, jacobi3d2d);
            cost += error_.squaredNorm();
            H += jacobi3d2d.transpose() * jacobi3d2d;
            b += -jacobi3d2d.transpose() * error_;
        }

        Matrix<double, 6, 1> dx;
        dx = H.ldlt().solve(b);

        if (isnan(dx[0]))
        {
            cout << "result is nan" << endl;
            break;
        }
        if (iter > 0 && cost >= lastcost)
        {
            cout << "cost:" << cost << ",lastcost:" << lastcost << endl;
            break;
        }

        pose_3d2d = Sophus::SE3d::exp(dx) * pose_3d2d;
        lastcost = cost;
        cout << "iterations " << iter << " cost:" << cost << " lastcost:" << lastcost << endl;
        if (dx.norm() < 1e-25)
        {
            break;
        }
        cout << "pose:\n"
             << pose_3d2d.matrix() << endl;
    }
    Matrix3d real_rotation;
    Vector3d real_translation(-0.127225965886,-0.00750729768072,0.0613858418151);
    real_rotation << 0.99786620258, -0.0516724160901, 0.0399124437155,
        0.050595891549, 0.998339762774, 0.02752769194,
        -0.0412686019426, -0.0254495477483, 0.998823919924;

    Matrix3d iter_rota = pose_3d2d.matrix().block<3, 3>(0, 0);
    Vector3d iter_translation = pose_3d2d.matrix().block<3,1>(0,3);

 
    Utility::Rotation_matrix_comparison(real_rotation,iter_rota,real_translation,iter_translation);
    
}
