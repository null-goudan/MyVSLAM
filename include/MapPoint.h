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

        void SetWorldPos(const cv::Mat &Pos);
        cv::Mat GetWorldPos();

        cv::Mat GetNormal();
        KeyFrame* GetReferenceKeyFrame();

        std::map<KeyFrame*, size_t> GetObservations();
        int Observations();

        void AddObservation(KeyFrame* pKF, size_t idx);
        void EraseObservation(KeyFrame* pKF);

        int GetIndexInKeyFrame(KeyFrame* pKF);
        bool IsInKeyFrame(KeyFrame* pKF);

        void ComputeDistinctiveDescriptors();

        cv::Mat GetDescriptor();

        void UpdateNormalAndDepth();

        void SetBadFlag();
        bool isBad();

    public:
        long unsigned int mnId;         // 这个点的全局ID
        static long unsigned int nNextId;
        const long int mnFirstKFid;     //创建该点的关键帧ID
        const long int mnFirstFrame;    // 创建该点的帧ID
        int nObs;

        // Tracking 用到的变量
        // :TODO

        // Local Mapping 用到的变量
        // :TODO

        // 回环检测 用到的变量
        // :TODO

        static std::mutex mGlobalMutex;

    protected:
        cv::Mat mWorldPos;      // MapPoint 在世界坐标系下的坐标 
        
        std::map<KeyFrame*,size_t> mObservations;  // 观测到该点的KeyFrame和该点在KF中的索引
        
        cv::Mat mNormalVector;      // 该点的平均观测方向的向量

        cv::Mat mDescriptor;        // 该点对应的描述子  通过 ComputeDistinctiveDescriptors() 得到的最优描述子

        KeyFrame* mpRefKF;      // 引用此点的关键帧

        // Bad flag (we do not currently erase MapPoint from memory)
        bool mbBad;
        MapPoint* mpReplaced;

        // Scale invariance distances
        float mfMinDistance;
        float mfMaxDistance;

        Map* mpMap;

        std::mutex mMutexPos;
        std::mutex mMutexFeatures;
    };
}

#endif