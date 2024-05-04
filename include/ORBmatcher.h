#ifndef ORBMATCHER_H
#define ORBMATCHER_H

#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "MapPoint.h"
#include "KeyFrame.h"
#include "Frame.h"

namespace Goudan_SLAM
{
    class ORBmatcher
    {
    public:
        ORBmatcher(float nnratio = 0.6, bool checkOri = true);
        static int DescriptorDistance(const cv::Mat &a, const cv::Mat &b);

        int SearchForInitialization(Frame &F1, Frame &F2, std::vector<cv::Point2f> &vbPrevMatched, std::vector<int> &vnMatches12, int windowSize = 10);
        /**
         * @brief 通过词包，对关键帧的特征点进行跟踪
         *
         * KeyFrame中包含了MapPoints，对这些MapPoints进行tracking \n
         * 由于每一个MapPoint对应有描述子，因此可以通过描述子距离进行跟踪 \n
         * 为了加速匹配过程，将关键帧和当前帧的描述子划分到特定层的nodes中 \n
         * 对属于同一node的描述子计算距离进行匹配 \n
         * 通过距离阈值、比例阈值和角度投票进行剔除误匹配
         * @param  pKF               KeyFrame
         * @param  F                 Current Frame
         * @param  vpMapPointMatches F中MapPoints对应的匹配，NULL表示未匹配
         * @return                   成功匹配的数量
         */
        int SearchByBoW(KeyFrame *pKF, Frame &F, std::vector<MapPoint *> &vpMapPointMatches);
        int SearchByBoW(KeyFrame *pKF1, KeyFrame *pKF2, std::vector<MapPoint *> &vpMatches12);

        // 通过投影，对Local MapPoint进行追踪， 将LocalMapPoint投影到当前帧中，由此增加当前帧的MapPoints  在Tracking线程中使用
        int SearchByProjection(Frame &F, const std::vector<MapPoint *> &vpMapPoints, const float th = 3);
        int SearchByProjection(Frame &CurrentFrame, const Frame &LastFrame, const float th);

        // Project MapPoints seen in KeyFrame into the Frame and search matches.
        // Used in relocalisation (Tracking)
        int SearchByProjection(Frame &CurrentFrame, KeyFrame *pKF, const std::set<MapPoint *> &sAlreadyFound, const float th, const int ORBdist);

        // Project MapPoints using a Similarity Transformation and search matches.
        // Used in loop detection (Loop Closing)
        int SearchByProjection(KeyFrame *pKF, cv::Mat Scw, const std::vector<MapPoint *> &vpPoints, std::vector<MapPoint *> &vpMatched, int th);

        // 对三角化进行匹配，并验证对极约束
        int SearchForTriangulation(KeyFrame *pKF1, KeyFrame *pKF2, cv::Mat F12,
                                   std::vector<pair<size_t, size_t>> &vMatchedPairs);

        // Search matches between MapPoints seen in KF1 and KF2 transforming by a Sim3 [s12*R12|t12]
        // In the stereo and RGB-D case, s12=1
        int SearchBySim3(KeyFrame *pKF1, KeyFrame *pKF2, std::vector<MapPoint *> &vpMatches12, const float &s12, const cv::Mat &R12, const cv::Mat &t12, const float th);

        // Project MapPoints into KeyFrame and search for duplicated MapPoints.
        int Fuse(KeyFrame *pKF, const vector<MapPoint *> &vpMapPoints, const float th = 3.0);

        // Project MapPoints into KeyFrame using a given Sim3 and search for duplicated MapPoints.
        int Fuse(KeyFrame *pKF, cv::Mat Scw, const std::vector<MapPoint *> &vpPoints, float th, vector<MapPoint *> &vpReplacePoint);

    public:
        static const int TH_LOW;
        static const int TH_HIGH;
        static const int HISTO_LENGTH;

    protected:
        bool CheckDistEpipolarLine(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const cv::Mat &F12, const KeyFrame *pKF);

        float RadiusByViewingCos(const float &viewCos);
        void ComputeThreeMaxima(std::vector<int> *histo, const int L, int &ind1, int &ind2, int &ind3);

        float mfNNratio;
        bool mbCheckOrientation;
    };
}

#endif