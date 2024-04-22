#ifndef CONVERTER_H
#define CONVERTER_H

#include <opencv2/core/core.hpp>
#include <Eigen/Dense>
#include"Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"
#include"Thirdparty/g2o/g2o/types/types_seven_dof_expmap.h"

// 提供一些常见的转换
// g2o  和 Eigen  cv:: Mat
namespace Goudan_SLAM{
class Converter
{
    public:
    //一个描述子矩阵到一串单行的描述子向量
    static std::vector<cv::Mat> toDescriptorVector(const cv::Mat &Descriptors);
    
    // g2o::SE3Quat to cv::Mat ;
    static cv::Mat toCvMat(const g2o::SE3Quat &SE3);

    static cv::Mat toCvMat(const Eigen::Matrix<double,4,4> &m);
    
    // cv::Mat to g2o::SE3Quat;
    static g2o::SE3Quat toSE3Quat(const cv::Mat &cvT);

};
}

#endif