#include "LocalMapping.h"
#include "ORBmatcher.h"
#include "Optimizer.h"

#include <mutex>

namespace Goudan_SLAM
{
    LocalMapping::LocalMapping(Map *pMap) : mbResetRequested(false), mbFinishRequested(false), mbFinished(true), mpMap(pMap),
                                            mbAbortBA(false), mbStopped(false), mbStopRequested(false), mbNotStop(false), mbAcceptKeyFrames(true)
    {
    }

    void LocalMapping::SetTracker(Tracking *pTracker)
    {
        mpTracker = pTracker;
    }

    // void LocalMapping::Run()
    // {
    //     mbFinished = false;
    //     while (1)
    //     {
    //         // 告诉Tracking, LocalMapping正处于繁忙状态
    //         SetAcceptKeyFrames(false);

    //         // 等待处理关键帧列表不为空
    //         if (CheckNewKeyFrames())
    //         {
    //             // 计算关键帧特征点Bow映射， 将关键帧插入局部地图
    //             ProcessNewKeyFrame();

    //             // 剔除ProcessNewFrame函数中引入的不合格的MapPoints
    //             MapPointCulling();

    //             // 相机运动中与相邻关键帧通过三角化恢复出一些MapPoints
    //             CreateNewMapPoints();

    //             // 已经处理完队列中的最后一个关键帧
    //             if (!CheckNewKeyFrames())
    //             {
    //                 // 检查并融合当前关键帧与相邻帧（两级相邻）重复的MapPoints
    //                 SearchInNeighbors();
    //             }

    //             mbAbortBA = false;

    //             // 已经处理完队列中的最后一个关键帧，并且闭环检测没有请求停止LocalMapping
    //             // 已经处理完队列中的最后的一个关键帧，并且闭环检测没有请求停止LocalMapping
    //             if (!CheckNewKeyFrames() && !stopRequested())
    //             {
    //                 // VI-D Local BA
    //                 if (mpMap->KeyFramesInMap() > 2)
    //                     Optimizer::LocalBundleAdjustment(mpCurrentKeyFrame, &mbAbortBA, mpMap);

    //                 // Check redundant local Keyframes
    //                 // VI-E local keyframes culling
    //                 // 检测并剔除当前帧相邻的关键帧中冗余的关键帧
    //                 // 剔除的标准是：该关键帧的90%的MapPoints可以被其它关键帧观测到
    //                 // trick!
    //                 // Tracking中先把关键帧交给LocalMapping线程
    //                 // 并且在Tracking中InsertKeyFrame函数的条件比较松，交给LocalMapping线程的关键帧会比较密
    //                 // 在这里再删除冗余的关键帧
    //                 KeyFrameCulling();
    //             }

    //             // 将当前帧加入到闭环检测队列中
    //             // mpLoopCloser->InsertKeyFrame(mpCurrentKeyFrame);
    //         }
    //         else if (Stop())
    //         {
    //             // Safe area to stop
    //             while (isStopped() && !CheckFinish())
    //             {
    //                 // usleep(3000);
    //                 std::this_thread::sleep_for(std::chrono::milliseconds(3));
    //             }
    //             if (CheckFinish())
    //                 break;
    //         }

    //         ResetIfRequested();

    //         // Tracking will see that Local Mapping is not busy
    //         SetAcceptKeyFrames(true);

    //         if (CheckFinish())
    //             break;

    //         // usleep(3000);
    //         std::this_thread::sleep_for(std::chrono::milliseconds(3));
    //     }
    // }

    /**
     * @brief 插入关键帧
     *
     * 将关键帧插入到地图中，以便将来进行局部地图优化
     * 这里仅仅是将关键帧插入到列表中进行等待
     * @param pKF KeyFrame
     */
    void LocalMapping::InsertKeyFrame(KeyFrame *pKF)
    {
        unique_lock<mutex> lock(mMutexNewKFs);
        // 将关键帧插入到列表中
        mlNewKeyFrames.push_back(pKF);
        mbAbortBA=true;
    }

    /**
     * @brief 查看列表中是否有等待被插入的关键帧
     * @return 如果存在，返回true
     */
    bool LocalMapping::CheckNewKeyFrames()
    {
        unique_lock<mutex> lock(mMutexNewKFs);
        return(!mlNewKeyFrames.empty());
    }


}