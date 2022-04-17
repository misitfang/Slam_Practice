#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include "utility.h"

using namespace Eigen;
using namespace std;

void Utility::Vector3d_Skew_Symmetric_Matrix(const Vector3d &vector_a, Matrix3d &Skew_Symmetric_Matrix)
{
     Matrix3d temp_a;
     temp_a << 0, -vector_a.row(2), vector_a.row(1), vector_a.row(2), 0, -vector_a.row(0), -vector_a.row(1), vector_a.row(0), 0;
     Skew_Symmetric_Matrix = temp_a;
}

void Utility::autoDiffJacobi(const Vector3d &point1, const Vector3d &point2, const Matrix3d &R,
                             const Matrix<double, 3, 1> &t,
                             Matrix<double, 3, 6> &J)

{

     // cout << "test^:\n"
     //      << Test << endl;
     Matrix<double, 3, 6> J1;
     double delta = 1e-10;
     for (int i = 0; i < 6; i++)
     {
          Matrix<double, 6, 1> delta_x = Matrix<double, 6, 1>::Zero();
          // Matrix<double,6,1>delta_x=Matrix::Zero();错误写法
          delta_x[i] = delta;

          // 向量分块操作vector;segment(i,j获取第i个元素开始的j个元素)
          Matrix<double, 3, 1> delta_theta = delta_x.segment(0, 3);
          Matrix<double, 3, 1> delta_t = delta_x.segment(3, 3);

          // 矩阵分块A.block<p,q矩阵大小>(i,j起始行列)方法2
          // Matrix<double,3,1> delta_theta = delta_x.block<3,1>(0,0);
          // Matrix<double,3,1> delta_t = delta_x.block<3,1>(3,0);s

          // 方法3
          // Matrix<double,3,1> delta_theta ;
          // delta_theta.row(0)=delta_x.row(0);
          // delta_theta.row(1)=delta_x.row(1);
          // delta_theta.row(2)=delta_x.row(2);
          // Matrix<double,3,1> delta_t;
          // delta_t.row(0)=delta_x.row(3);
          // delta_t.row(1)=delta_x.row(4);
          // delta_t.row(2)=delta_x.row(5);

          // delta_theta的反对称矩阵Matrix<double, 3, 3> delta_x_hat = Sophus::SO3d::hat(delta_theta);
          Matrix3d delta_x_hat;
          Utility::Vector3d_Skew_Symmetric_Matrix(delta_theta,delta_x_hat);

          // J1.transpose()[i]=((point1-exp(Sophus::SO3d::hat(delta_x))*R*point2-(t+delta_x))-(point1-R*point2-t))/delta_x;
          J1.col(i) = ((point1 - (Matrix3d::Identity() + delta_x_hat) * R * point2 - (t + delta_t)) - (point1 - R * point2 - t)) / delta; //p-(I+φ^)*R*p2-......
     }
     // cout<<"J1:\n"<<J1<<endl;
     Matrix<double, 3, 6> J_J1 = J - J1;
     double delta_J_J1;
     delta_J_J1 = J_J1.norm();
     cout << "delta_J_J1:\n"
          << delta_J_J1 << endl;

     cout << "1" << endl;
};

void Utility::Rotation_matrix_comparison(const Matrix3d &real_rotation_matrix,
                                         const Matrix3d &my_rotation_matrix,
                                         const Vector3d &real_translation,
                                         const Vector3d &my_translation)
{
     Quaterniond real_rot_qua(real_rotation_matrix);
     Quaterniond my_rot_qua(my_rotation_matrix);
     double temp_R = ((real_rot_qua.inverse() * my_rot_qua).vec()).norm();
     cout << "the error of real_rotation_matrix and my_rotation_matrix:\n"
          << temp_R << endl;

     double temp_t = (real_translation - my_translation).norm();
     cout << "the error of real_translation_matrix and my translation_matrix:\n"
          << temp_t << endl;
}

//