#include "Converter.h"

namespace Goudan_SLAM
{
    std::vector<cv::Mat> Converter::toDescriptorVector(const cv::Mat &Descriptors)
    {
        std::vector<cv::Mat> vDesc;
        vDesc.reserve(Descriptors.rows);
        for (int j = 0; j < Descriptors.rows; j++)
            vDesc.push_back(Descriptors.row(j));

        return vDesc;
    }

    g2o::SE3Quat Converter::toSE3Quat(const cv::Mat &cvT)
    {
        Eigen::Matrix<double, 3, 3> R;
        R << cvT.at<float>(0, 0), cvT.at<float>(0, 1), cvT.at<float>(0, 2),
            cvT.at<float>(1, 0), cvT.at<float>(1, 1), cvT.at<float>(1, 2),
            cvT.at<float>(2, 0), cvT.at<float>(2, 1), cvT.at<float>(2, 2);

        Eigen::Matrix<double, 3, 1> t(cvT.at<float>(0, 3), cvT.at<float>(1, 3), cvT.at<float>(2, 3));

        return g2o::SE3Quat(R, t);
    }

    cv::Mat Converter::toCvMat(const g2o::SE3Quat &SE3)
    {
        Eigen::Matrix<double, 4, 4> eigMat = SE3.to_homogeneous_matrix();
        return toCvMat(eigMat);
    }

    cv::Mat Converter::toCvMat(const Eigen::Matrix<double, 4, 4> &m)
    {
        cv::Mat cvMat(4, 4, CV_32F);
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                cvMat.at<float>(i, j) = m(i, j);

        return cvMat.clone();
    }
}