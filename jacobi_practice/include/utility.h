#ifndef UTILITY_H_
#define UTILITY_H_
#include <Eigen/Core>

using namespace Eigen;

class Utility
{
public:

    static void Vector3d_Skew_Symmetric_Matrix(const Vector3d &vector_a, Matrix3d &Skew_Symmetric_Matrix);

    static void autoDiffJacobi(const Vector3d &point1,
                               const Vector3d &point2,
                               const Matrix3d &R,
                               const Matrix<double, 3, 1> &t,
                               Matrix<double, 3, 6> &J);

    static void Rotation_matrix_comparison(const Matrix3d &real_rotation_matrix,
                                           const Matrix3d &my_rotation_matrix,
                                           const Vector3d &real_translation,
                                           const Vector3d &my_translatio);
};
#endif
