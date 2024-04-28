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
        int SearchByBoW(KeyFrame *pKF, Frame &F, vector<MapPoint *> &vpMapPointMatches);

        // 通过投影，对Local MapPoint进行追踪， 将LocalMapPoint投影到当前帧中，由此增加当前帧的MapPoints  在Tracking线程中使用
        int SearchByProjection(Frame &F, const std::vector<MapPoint *> &vpMapPoints, const float th = 3);
        int SearchByProjection(Frame &CurrentFrame, const Frame &LastFrame, const float th);

        // Project MapPoints into KeyFrame and search for duplicated MapPoints.
        int Fuse(KeyFrame *pKF, const vector<MapPoint *> &vpMapPoints, const float th = 3.0);

        // 对三角化进行匹配，并验证对极约束
        int SearchForTriangulation(KeyFrame *pKF1, KeyFrame *pKF2, cv::Mat F12,
                                   std::vector<pair<size_t, size_t>> &vMatchedPairs);

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