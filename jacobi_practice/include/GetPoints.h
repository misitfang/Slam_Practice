#ifndef GETPOINTS_H_
#define GETPOINTS_H
#include <Eigen/Core>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace Eigen;

class GetPoints
{
public:
    static void find_feature_matches(const Mat &img_1, const Mat &img_2,
                              std::vector<KeyPoint> &keypoints_1,
                              std::vector<KeyPoint> &keypoints_2,
                              std::vector<DMatch> &matches);


};



#endif