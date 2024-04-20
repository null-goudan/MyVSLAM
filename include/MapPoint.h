#ifndef MAPPOINT_H
#define MAPPOINT_H

#include "KeyFrame.h"
#include "Frame.h"
#include "Map.h"

#include <opencv2/core/core.hpp>
#include <mutex>
#include <map>

namespace Goudan_SLAM
{
    class KeyFrame;
    class Map;
    class Frame;

    class MapPoint
    {
    public:
        MapPoint(const cv::Mat &Pos, KeyFrame *pRefKF, Map *pMap);
        MapPoint(const cv::Mat &Pos, Map *pMap, Frame *pFrame, const int &idxF);

        void SetWorldPos(const cv::Mat &Pos);
        cv::Mat GetWorldPos();

        // cv::Mat GetNormal();
        KeyFrame* GetReferenceKeyFrame();

        // void SetBadFlag();
        // bool isBad();

    public:
        long unsigned int mnId;
        static long unsigned int nNextId;
        const long int mnFirstKFid;     //创建该点的关键帧ID
        const long int mnFirstFrame;    // 创建该点的帧ID
        int nObs;

        static std::mutex mGlobalMutex;

    protected:
        cv::Mat mWorldPos;      // MapPoint 在世界坐标系下的坐标 
        
        std::map<KeyFrame*, size_t> mObservations; // 观测到该点的KeyFrame和该点在KF中的索引
        
        KeyFrame* mpRefKF;      // 引用此点的关键帧

        Map* mpMap;

        // Bad flag (we do not currently erase MapPoint from memory)
        bool mbBad;
        // MapPoint* mpReplaced;

        std::mutex mMutexPos;
        std::mutex mMutexFeatures;
    };
}

#endif