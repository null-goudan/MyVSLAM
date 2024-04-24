#ifndef CONVERTER_H
#define CONVERTER_H

#include <opencv2/core/core.hpp>
#include <Eigen/Dense>
#include "Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"
#include "Thirdparty/g2o/g2o/types/types_seven_dof_expmap.h"

// 提供一些常见的转换
// g2o  和 Eigen  cv:: Mat
namespace Goudan_SLAM
{
    class Converter
    {
    public:
        /**
         * @brief 一个描述子矩阵到一串单行的描述子向量
         */
        static std::vector<cv::Mat> toDescriptorVector(const cv::Mat &Descriptors);

        /**
         * @name toSE3Quat
         */
        ///@{
        /** cv::Mat to g2o::SE3Quat */
        static g2o::SE3Quat toSE3Quat(const cv::Mat &cvT);
        /** unimplemented */
        static g2o::SE3Quat toSE3Quat(const g2o::Sim3 &gSim3);
        ///@}

        /**
         * @name toCvMat
         */
        ///@{
        static cv::Mat toCvMat(const g2o::SE3Quat &SE3);
        static cv::Mat toCvMat(const g2o::Sim3 &Sim3);
        static cv::Mat toCvMat(const Eigen::Matrix<double, 4, 4> &m);
        static cv::Mat toCvMat(const Eigen::Matrix3d &m);
        static cv::Mat toCvMat(const Eigen::Matrix<double, 3, 1> &m);
        static cv::Mat toCvSE3(const Eigen::Matrix<double, 3, 3> &R, const Eigen::Matrix<double, 3, 1> &t);
        ///@}

        /**
         * @name toEigen
         */
        ///@{
        static Eigen::Matrix<double, 3, 1> toVector3d(const cv::Mat &cvVector);
        static Eigen::Matrix<double, 3, 1> toVector3d(const cv::Point3f &cvPoint);
        static Eigen::Matrix<double, 3, 3> toMatrix3d(const cv::Mat &cvMat3);
        static std::vector<float> toQuaternion(const cv::Mat &M);
        ///@}
    };
}

#endif