#include "MapPoint.h"

#include <mutex>

namespace Goudan_SLAM
{
    long unsigned int MapPoint::nNextId = 0;
    mutex MapPoint::mGlobalMutex;

    // 给定坐标和关键帧构造地图点
    MapPoint::MapPoint(const cv::Mat &Pos, KeyFrame *pRefKF, Map *pMap)
        : mnFirstKFid(pRefKF->mnID), mnFirstFrame(pRefKF->mnFrameId), mpMap(pMap), mpRefKF(pRefKF)
    {
        Pos.copyTo(mWorldPos);

        unique_lock<mutex> lock(mpMap->mMutexPointCreation);
        mnId = nNextId++;
    }

    // 给定坐标与frame构造MapPoint
    MapPoint::MapPoint(const cv::Mat &Pos, Map *pMap, Frame *pFrame, const int &idxF)
        : mnFirstKFid(-1), mnFirstFrame(pFrame->mnId), mpMap(pMap), mpRefKF(static_cast<KeyFrame *>(NULL))
    {
        Pos.copyTo(mWorldPos);

        unique_lock<mutex> lock(mpMap->mMutexPointCreation);
        mnId = nNextId++;
    }
    void MapPoint::SetWorldPos(const cv::Mat &Pos)
    {
        unique_lock<mutex> lock2(mGlobalMutex);
        unique_lock<mutex> lock(mMutexPos);
        Pos.copyTo(mWorldPos);
    }

    cv::Mat MapPoint::GetWorldPos()
    {
        unique_lock<mutex> lock(mMutexPos);
        return mWorldPos.clone();
    }

    // cv::Mat MapPoint::GetNormal()
    // {
    //     unique_lock<mutex> lock(mMutexPos);
    //     return mNormalVector.clone();
    // }

    KeyFrame *MapPoint::GetReferenceKeyFrame()
    {
        unique_lock<mutex> lock(mMutexFeatures);
        return mpRefKF;
    }
}