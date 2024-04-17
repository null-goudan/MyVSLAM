#ifndef INITIALIZER_H
#define INITIALIZER_H

#include <opencv2/opencv.hpp>
#include "Frame.h"

namespace Goudan_SLAM{
    // 单目SLAM初始化器
    class Initializer
    {
        typedef pair<int, int> Match;
    public:
        // 用Reference frame来初始化， 这个Reference Frame 就是 SLAM 正式开始的第一针
        Initializer(const Frame& ReferenceFrame, float sigma = 1.0, int iterations = 200);

        // 用current Frame, 也就是用SLAM逻辑上的第二帧来初始化整个SLAM， 得到最开始两帧之间的R t, 以及点云
        bool Initialize(const Frame &CurrentFrame, const vector<int> &vMatches12,
                    cv::Mat &R21, cv::Mat &t21, vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated);

    private:

        vector <cv::KeyPoint> mvKeys1;  ///< 存储Reference Frame中的特征点
        vector <cv::KeyPoint> mvKeys2;  ///< 存储Current Frame中的特征点

        vector<Match> mvMatches12;      ///< Match的数据结构是pair, mvMatch12只记录从Reference到Current的匹配点
        vector<bool> mvbMatches1;       ///< 记录Reference Frame的每个特征点在Current Frame是否有匹配的特征点

        cv::Mat mK; // 相机内参

        float mSigma, mSigma2;  //测量误差

        // 计算 Fundamental矩阵和Homography矩阵时RANSAC迭代次数
        int mMaxIterations;

        vector<vector<size_t> > mvSets; ///< 二维容器，外层容器的大小为迭代次数，内层容器大小为每次迭代算H或F矩阵需要的点
    };
}

#endif