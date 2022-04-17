#include <iostream>
#include "GetPoints.h"
#include "utility.h"
#include "icp3d3d_factor.h"
#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>

using namespace cv;

using namespace std;
using namespace Eigen;

typedef vector<Vector3d, Eigen::aligned_allocator<Vector3d>> Vecvector3d;



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
        cout << "usage: main img1 img2 depth1 depth2" << endl;
        return 1;
    }
    //-- 读取图像
    Mat img_1 = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    Mat img_2 = imread(argv[2], CV_LOAD_IMAGE_COLOR);

    vector<KeyPoint> keypoints_1, keypoints_2;
    vector<DMatch> matches;
    GetPoints::find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches);
    cout << "一共找到了" << matches.size() << "组匹配点" << endl;

    // 建立3D点
    Mat depth1 = imread(argv[3], CV_LOAD_IMAGE_UNCHANGED); 
    Mat depth2 = imread(argv[4], CV_LOAD_IMAGE_UNCHANGED); 
    Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    Vecvector3d points1, points2;

    for (DMatch m : matches)
    {
        ushort d1 = depth1.ptr<unsigned short>(int(keypoints_1[m.queryIdx].pt.y))[int(keypoints_1[m.queryIdx].pt.x)];
        ushort d2 = depth2.ptr<unsigned short>(int(keypoints_2[m.trainIdx].pt.y))[int(keypoints_2[m.trainIdx].pt.x)];
        if (d1 == 0 || d2 == 0) // bad depth
            continue;
        Point2d p1 = pixel2cam(keypoints_1[m.queryIdx].pt, K);
        Point2d p2 = pixel2cam(keypoints_2[m.trainIdx].pt, K);
        float dd1 = float(d1) / 5000.0;
        float dd2 = float(d2) / 5000.0;
        points1.push_back(Vector3d(p1.x * dd1, p1.y * dd1, dd1));
        points2.push_back(Vector3d(p2.x * dd2, p2.y * dd2, dd2));
    }
    Matrix3d R_1;
    R_1 << 0.9972395977366735, 0.05617039856770108, -0.04855997354553421,
        -0.05598345194682008, 0.9984181427731503, 0.005202431117422968,
        0.04877538122983249, -0.002469515369266706, 0.9988067198811419;
    Matrix<double, 3, 1> t_1;
    t_1 << 0.1417248739257467, -0.05551033302525168, -0.03119093188273836;
    Matrix<double, 3, 6> J_1;
    Vector3d error;

    const int iterations = 10;
    double cost = 0, lastcost = 0;
    for (int iter = 0; iter < iterations; iter++)
    {
        Matrix<double, 6, 6> H = Matrix<double, 6, 6>::Zero();
        Matrix<double, 6, 1> b = Matrix<double, 6, 1>::Zero();

        cost = 0;
        for (int i = 0; i < points1.size(); i++)
        {
            // Point3dJacobi p3d_p3d_error;
            //计算error
            Icp3d3d_factor::icp3d3d_error(points1[i], points2[i], R_1, t_1, error);
            // cout << "error" << i << ":\n"
            //      << error << endl;
            cost += error.squaredNorm();

            //计算雅克比
            Icp3d3d_factor::icp3d3d_jacobi(points1[i], points2[i], R_1, t_1, J_1);
            // cout << "jacobi" << i << ":\n"
            //      << J_1 << endl;

            H += J_1.transpose() * J_1;
            b += -J_1.transpose() * error;
            //数值求导验证雅克比求解是否正确
            // Utility::autoDiffJacobi(points1[i], points2[i], R_1, t_1, J_1);
        };
        Matrix<double, 6, 1> dx_;
        dx_ = H.ldlt().solve(b);
        cout << "dx_" << iter << ":\n"
             << dx_ << endl;

        if (isnan(dx_[0]))
        {
            cout << "result is nan" << endl;
            break;
        }
        if (iter > 0 && cost >= lastcost)
        {
            cout << "cost:" << cost << ",lastcost:" << lastcost << endl;
            break;
        }
        Matrix<double, 3, 1> delta_theta_dx = dx_.segment(0, 3);
        Matrix<double, 3, 1> delta_t_dx = dx_.segment(3, 3);
        // Utility::Vector3d_Skew_Symmetric_Matrix(delta_theta_dx,Matrix3d delta_theta_dx_hat);
        // Matrix3d delta_theta_dx_hat = Sophus::SO3d::hat(delta_theta_dx);
        AngleAxisd delta_rot_vec(delta_theta_dx.norm(), delta_theta_dx.normalized()); //
        Matrix3d RR = delta_rot_vec.matrix();
        // cout<<"RR:\n"<<RR<<endl;
        t_1 = t_1 + delta_t_dx;
        R_1 = RR * R_1;//旋转向量增量转换为旋转矩阵增量

        // R_1 = (Matrix3d::Identity() + delta_theta_dx_hat) * R_1;
        lastcost = cost;
        cout << "iterations " << iter << " cost:" << cost << " lastcost:" << lastcost << endl;
        if (dx_.norm() < 1e-25)
        {
            break;
        }
    }
    cout << "R:\n"
         << R_1.matrix() << "\n"
         << "t：\n"
         << t_1 << endl;
    Matrix3d real_R;
    real_R << 0.997239, 0.05617, -0.04855, -0.055998, 0.998418, 0.005202, 0.0487753, -0.002469, 0.998806;
    Quaterniond my_rot_qua(R_1);
    Quaterniond real_rot_qua(real_R);
    // cout << "quaternion from rotation matrixmy = " << my_rot_qua.coeffs().transpose() << endl;
    // cout << "quaternion from rotation matrixreal = " << real_rot_qua.coeffs().transpose() << endl;
    cout << " h " << ((real_rot_qua.inverse() * my_rot_qua).vec()).norm() << endl;
    Vector3d real_t(0.141, -0.055, -0.031);
    cout << (real_t - t_1).norm() << endl;

    for (int j = 0; j < 1; j++)
    {
        cout << "points1:\n"
             << points1[j].transpose() << endl;
        cout << "p1 = RP2 + t=:\n"
             << (R_1 * points2[j] + t_1).transpose() << endl;
    }
}

    