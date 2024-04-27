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

    void LocalMapping::Run()
    {
        mbFinished = false;
        while (1)
        {
            // 告诉Tracking, LocalMapping正处于繁忙状态
            SetAcceptKeyFrames(false);

            // 等待处理关键帧列表不为空
            if (CheckNewKeyFrames())
            {
                // 计算关键帧特征点Bow映射， 将关键帧插入局部地图
                ProcessNewKeyFrame();

                // 剔除ProcessNewFrame函数中引入的不合格的MapPoints
                MapPointCulling();

                // 相机运动中与相邻关键帧通过三角化恢复出一些MapPoints
                CreateNewMapPoints();

                // 已经处理完队列中的最后一个关键帧
                if (!CheckNewKeyFrames())
                {
                    // 检查并融合当前关键帧与相邻帧（两级相邻）重复的MapPoints
                    SearchInNeighbors();
                }

                mbAbortBA = false;

                // 已经处理完队列中的最后一个关键帧，并且闭环检测没有请求停止LocalMapping
                // 已经处理完队列中的最后的一个关键帧，并且闭环检测没有请求停止LocalMapping
                if (!CheckNewKeyFrames() && !stopRequested())
                {
                    // VI-D Local BA
                    if (mpMap->KeyFramesInMap() > 2)
                        Optimizer::LocalBundleAdjustment(mpCurrentKeyFrame, &mbAbortBA, mpMap);

                    // Check redundant local Keyframes
                    // VI-E local keyframes culling
                    // 检测并剔除当前帧相邻的关键帧中冗余的关键帧
                    // 剔除的标准是：该关键帧的90%的MapPoints可以被其它关键帧观测到
                    // trick!
                    // Tracking中先把关键帧交给LocalMapping线程
                    // 并且在Tracking中InsertKeyFrame函数的条件比较松，交给LocalMapping线程的关键帧会比较密
                    // 在这里再删除冗余的关键帧
                    KeyFrameCulling();
                }

                // 将当前帧加入到闭环检测队列中
                // mpLoopCloser->InsertKeyFrame(mpCurrentKeyFrame);
            }
            else if (Stop())
            {
                // Safe area to stop
                while (isStopped() && !CheckFinish())
                {
                    // usleep(3000);
                    std::this_thread::sleep_for(std::chrono::milliseconds(3));
                }
                if (CheckFinish())
                    break;
            }

            ResetIfRequested();

            // Tracking will see that Local Mapping is not busy
            SetAcceptKeyFrames(true);

            if (CheckFinish())
                break;

            // usleep(3000);
            std::this_thread::sleep_for(std::chrono::milliseconds(3));
        }
    }

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
        mbAbortBA = true;
    }

    /**
     * @brief 查看列表中是否有等待被插入的关键帧
     * @return 如果存在，返回true
     */
    bool LocalMapping::CheckNewKeyFrames()
    {
        unique_lock<mutex> lock(mMutexNewKFs);
        return (!mlNewKeyFrames.empty());
    }

    bool LocalMapping::AcceptKeyFrames()
    {
        unique_lock<mutex> lock(mMutexAccept);
        return mbAcceptKeyFrames;
    }

    void LocalMapping::SetAcceptKeyFrames(bool flag)
    {
        unique_lock<mutex> lock(mMutexAccept);
        mbAcceptKeyFrames = flag;
    }

    /**
     * @brief 处理列表中的关键帧
     *
     * - 计算Bow，加速三角化新的MapPoints
     * - 关联当前关键帧至MapPoints，并更新MapPoints的平均观测方向和观测距离范围
     * - 插入关键帧，更新Covisibility图和Essential图
     * @see VI-A keyframe insertion
     */
    void LocalMapping::ProcessNewKeyFrame()
    {
        // 1. 从缓冲队列取出一帧关键帧
        {
            unique_lock<mutex> lock(mMutexNewKFs);
            // 从列表中获得一个等待被插入的关键帧
            mpCurrentKeyFrame = mlNewKeyFrames.front();
            mlNewKeyFrames.pop_front();
        }

        // 2. 计算改关键帧特征点的Bow映射关系
        mpCurrentKeyFrame->ComputeBoW();

        // 3. 跟踪局部地图过程中新匹配上的MapPoints和当前关键帧绑定
        // 在TrackLocalMap函数中将局部地图中的MapPoints与当前帧进行了匹配
        // 但没有对这些匹配上的MapPoints与当前帧进行关联
        const vector<MapPoint *> vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();

        for (size_t i = 0; i < vpMapPointMatches.size(); i++)
        {
            MapPoint *pMP = vpMapPointMatches[i];
            if (pMP)
            {
                if (!pMP->isBad())
                {
                    // 非当前帧生成的MapPoints
                    // 为当前帧在tracking过程跟踪到的MapPoints更新属性
                    if (!pMP->IsInKeyFrame(mpCurrentKeyFrame))
                    {
                        // 添加观测
                        pMP->AddObservation(mpCurrentKeyFrame, i);
                        // 获得该点的平均观测方向和观测距离范围
                        pMP->UpdateNormalAndDepth();
                        // 加入关键帧后，更新3d点的最佳描述子
                        pMP->ComputeDistinctiveDescriptors();
                    }
                }
            }
        }
        // 4. 更新关键帧间的连接关系，Covisibility图和Essential图(tree)
        mpCurrentKeyFrame->UpdateConnections();
        // 5. 将该关键帧插入到地图中
        mpMap->AddKeyFrame(mpCurrentKeyFrame);
    }

    /**
     * @brief 剔除ProcessNewKeyFrame和CreateNewMapPoints函数中引入的质量不好的MapPoints
     * @see VI-B recent map points culling
     */
    void LocalMapping::MapPointCulling()
    {
        // Check Recent Added MapPoints
        list<MapPoint *>::iterator lit = mlpRecentAddedMapPoints.begin();
        const unsigned long int nCurrentKFid = mpCurrentKeyFrame->mnID;

        int nThObs = 2;
        const int cnThObs = nThObs;

        // 遍历等待检查的MapPoints
        while (lit != mlpRecentAddedMapPoints.end())
        {
            MapPoint *pMP = *lit;
            if (pMP->isBad())
            {
                // 步骤1：已经是坏点的MapPoints直接从检查链表中删除
                lit = mlpRecentAddedMapPoints.erase(lit);
            }
            else if (pMP->GetFoundRatio() < 0.25f)
            {
                // 步骤2：将不满足VI-B条件的MapPoint剔除
                // VI-B 条件1：
                // 跟踪到该MapPoint的Frame数相比预计可观测到该MapPoint的Frame数的比例需大于25%
                // IncreaseFound / IncreaseVisible < 25%，注意不一定是关键帧。
                pMP->SetBadFlag();
                lit = mlpRecentAddedMapPoints.erase(lit);
            }
            else if (((int)nCurrentKFid - (int)pMP->mnFirstKFid) >= 2 && pMP->Observations() <= cnThObs)
            {
                // 步骤3：将不满足VI-B条件的MapPoint剔除
                // VI-B 条件2：从该点建立开始，到现在已经过了不小于2个关键帧
                // 但是观测到该点的关键帧数却不超过cnThObs帧，那么该点检验不合格
                pMP->SetBadFlag();
                lit = mlpRecentAddedMapPoints.erase(lit);
            }
            else if (((int)nCurrentKFid - (int)pMP->mnFirstKFid) >= 3)
                // 步骤4：从建立该点开始，已经过了3个关键帧而没有被剔除，则认为是质量高的点
                // 因此没有SetBadFlag()，仅从队列中删除，放弃继续对该MapPoint的检测
                lit = mlpRecentAddedMapPoints.erase(lit);
            else
                lit++;
        }
    }

    /**
     * 相机运动过程中和共视程度比较高的关键帧通过三角化恢复出一些MapPoints
     */
    void LocalMapping::CreateNewMapPoints()
    {
        int nn = 20;

        // 1. 在当前关键帧的共视关键帧中找到共视程度最高的nn帧相邻帧vpNeighKFs
        const vector<KeyFrame*> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);

        
    }
}