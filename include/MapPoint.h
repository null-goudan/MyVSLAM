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
        MapPoint(const cv::Mat &Pos,  Map* pMap, Frame* pFrame, const int &idxF);

        void SetWorldPos(const cv::Mat &Pos);
        cv::Mat GetWorldPos();

        cv::Mat GetNormal();
        KeyFrame *GetReferenceKeyFrame();

        std::map<KeyFrame *, size_t> GetObservations();
        int Observations();

        void AddObservation(KeyFrame *pKF, size_t idx);
        void EraseObservation(KeyFrame *pKF);

        int GetIndexInKeyFrame(KeyFrame *pKF);
        bool IsInKeyFrame(KeyFrame *pKF);

        void SetBadFlag();
        bool isBad();

        void Replace(MapPoint *pMP);
        MapPoint *GetReplaced();

        void IncreaseVisible(int n = 1);
        void IncreaseFound(int n = 1);
        float GetFoundRatio();
        inline int GetFound()
        {
            return mnFound;
        }

        void ComputeDistinctiveDescriptors();

        cv::Mat GetDescriptor();

        void UpdateNormalAndDepth();

        float GetMinDistanceInvariance();
        float GetMaxDistanceInvariance();
        int PredictScale(const float &currentDist, KeyFrame *pKF);
        int PredictScale(const float &currentDist, Frame *pF);

    public:
        long unsigned int mnId; // 这个点的全局ID
        static long unsigned int nNextId;
        const long int mnFirstKFid;  // 创建该点的关键帧ID
        const long int mnFirstFrame; // 创建该点的帧ID
        int nObs;

        // Tracking 用到的变量
        float mTrackProjX;
        float mTrackProjY;
        float mTrackProjXR;
        int mnTrackScaleLevel;
        float mTrackViewCos;
        // TrackLocalMap - SearchByProjection中决定是否对该点进行投影的变量
        // mbTrackInView==false的点有几种：
        // a 已经和当前帧经过匹配（TrackReferenceKeyFrame，TrackWithMotionModel）但在优化过程中认为是外点
        // b 已经和当前帧经过匹配且为内点，这类点也不需要再进行投影
        // c 不在当前相机视野中的点（即未通过isInFrustum判断）
        bool mbTrackInView;
        // TrackLocalMap - UpdateLocalPoints中防止将MapPoints重复添加至mvpLocalMapPoints的标记
        long unsigned int mnTrackReferenceForFrame;
        // TrackLocalMap - SearchLocalPoints中决定是否进行isInFrustum判断的变量
        // mnLastFrameSeen==mCurrentFrame.mnId的点有几种：
        // a 已经和当前帧经过匹配（TrackReferenceKeyFrame，TrackWithMotionModel）但在优化过程中认为是外点
        // b 已经和当前帧经过匹配且为内点，这类点也不需要再进行投影
        long unsigned int mnLastFrameSeen;

        // Local Mapping 用到的变量
        long unsigned int mnBALocalForKF;
        long unsigned int mnFuseCandidateForKF;

        // 回环检测 用到的变量
        long unsigned int mnLoopPointForKF;
        long unsigned int mnCorrectedByKF;
        long unsigned int mnCorrectedReference;
        cv::Mat mPosGBA;
        long unsigned int mnBAGlobalForKF;

        static std::mutex mGlobalMutex;

    protected:
        cv::Mat mWorldPos; // MapPoint 在世界坐标系下的坐标

        std::map<KeyFrame *, size_t> mObservations; // 观测到该点的KeyFrame和该点在KF中的索引

        cv::Mat mNormalVector; // 该点的平均观测方向的向量

        cv::Mat mDescriptor; // 该点对应的描述子  通过 ComputeDistinctiveDescriptors() 得到的最优描述子

        KeyFrame *mpRefKF; // 引用此点的关键帧

        // Tracking counters
        int mnVisible;
        int mnFound;

        // Bad flag (we do not currently erase MapPoint from memory)
        bool mbBad;
        MapPoint *mpReplaced;

        // Scale invariance distances
        float mfMinDistance;
        float mfMaxDistance;

        Map *mpMap;

        std::mutex mMutexPos;
        std::mutex mMutexFeatures;
    };
}

#endif