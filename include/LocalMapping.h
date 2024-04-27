#ifndef LOCALMAPPING_H
#define LOCALMAPPING_H

#include "KeyFrame.h"
#include "Map.h"
#include "Tracking.h"
#include "KeyFrameDatabase.h"

#include <mutex>

namespace Goudan_SLAM
{
    class Tracking;
    class Map;

    class LocalMapping
    {
    public:
        LocalMapping(Map *pMap);

        void SetTracker(Tracking *pTracker);

        void Run();

        void InsertKeyFrame(KeyFrame *pKF);

        // Tread Synch
        void RequestStop();
        void RequestReset();
        bool Stop();
        void Release();
        bool isStopped();
        bool stopRequested();
        bool AcceptKeyFrames();
        void SetAcceptKeyFrames(bool flag);
        bool SetNotStop(bool flag);

        void InterruptBA();

        void RequestFinish();
        bool isFinished();

        int KeyframesInQueue()
        {
            unique_lock<std::mutex> lock(mMutexNewKFs);
            return mlNewKeyFrames.size();
        }

    protected:
        bool CheckNewKeyFrames();
        void ProcessNewKeyFrame();
        void CreateNewMapPoints();

        void MapPointCulling();
        void SearchInNeighbors();

        void KeyFrameCulling();

        cv::Mat ComputeF12(KeyFrame *&pKF1, KeyFrame *&pKF2);

        cv::Mat SkewSymmetricMatrix(const cv::Mat &v);


        void ResetIfRequested();
        bool mbResetRequested;
        std::mutex mMutexReset;

        bool CheckFinish();
        void SetFinish();
        bool mbFinishRequested;
        bool mbFinished;
        std::mutex mMutexFinish;

        Map *mpMap;

        // LoopClosing *mpLoopCloser;
        Tracking *mpTracker;

        // Tracking线程向LocalMapping中插入关键帧是先插入到该队列中
        std::list<KeyFrame *> mlNewKeyFrames; ///< 等待处理的关键帧列表

        KeyFrame *mpCurrentKeyFrame;

        std::list<MapPoint *> mlpRecentAddedMapPoints;

        std::mutex mMutexNewKFs;

        bool mbAbortBA;

        bool mbStopped;
        bool mbStopRequested;
        bool mbNotStop;
        std::mutex mMutexStop;

        bool mbAcceptKeyFrames;
        std::mutex mMutexAccept;
    };
}

#endif