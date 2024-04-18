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
        // 假设场景为平面情况下通过前两帧求取Homography矩阵(current frame 2 到 reference frame 1),并得到该模型的评分
        void FindHomography(vector<bool> &vbMatchesInliers, float &score, cv::Mat &H21);
        // 假设场景为非平面情况下通过前两帧求取Fundamental矩阵(current frame 2 到 reference frame 1),并得到该模型的评分
        void FindFundamental(vector<bool> &vbInliers, float &score, cv::Mat &F21);

        // 具体计算Homography矩阵
        cv::Mat ComputeH21(const vector<cv::Point2f> &vP1, const vector<cv::Point2f> &vP2);

        // 具体计算Fundamental矩阵
        cv::Mat ComputeF21(const vector<cv::Point2f> &vP1, const vector<cv::Point2f> &vP2);

        // 具体计算假设使用Homography模型的得分
        float CheckHomography(const cv::Mat &H21, const cv::Mat H12,vector<bool> vbMatchesInliners, float sigma);

        // 具体计算假设使用Fundamental模型的得分
        float CheckFundamental(const cv::Mat &F21, vector<bool> &bvMatcherInliners, float sigma);

        // 分解F矩阵，并从分解后的多个解中找出合适的R，t
        bool ReconstructF(vector<bool> &vbMatchesInliers, cv::Mat &F21, cv::Mat &K,
                        cv::Mat &R21, cv::Mat &t21, vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated, float minParallax, int minTriangulated);

        // 分解H矩阵，并从分解后的多个解中找出合适的R，t
        bool ReconstructH(vector<bool> &vbMatchesInliers, cv::Mat &H21, cv::Mat &K,
                        cv::Mat &R21, cv::Mat &t21, vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated, float minParallax, int minTriangulated);
        
        // F矩阵通过结合内参可以得到Essential矩阵，该函数用于分解E矩阵，将得到4组解
        void DecomposeE(const cv::Mat &E, cv::Mat &R1, cv::Mat &R2, cv::Mat &t);
        
        // ReconstructF 调用该函数进行 cheirality check, 从而进一步找出F分解后最合适的解
        int CheckRT(const cv::Mat &R, const cv::Mat &t,const vector<cv::KeyPoint> &vKeys1, const vector<cv::KeyPoint> &vKeys2,
                       const vector<Match> &vMatches12, vector<bool> &vbInliers,
                       const cv::Mat &K, vector<cv::Point3f> &vP3D, float th2, vector<bool> &vbGood, float &parallax);

        // 归一化三维空间点和帧间位移t
        void Normalize(const vector<cv::KeyPoint> &vKeys, vector<cv::Point2f> &vNormalizedPoints, cv::Mat &T);

        // 通过三角化方法，利用反投影矩阵将特征点恢复为3D点
        void Triangulate(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const cv::Mat &P1, const cv::Mat &P2, cv::Mat &x3D);

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