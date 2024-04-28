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
        const vector<KeyFrame *> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);

        ORBmatcher matcher(0.6, false);

        cv::Mat Rcw1 = mpCurrentKeyFrame->GetRotation();
        cv::Mat Rwc1 = Rcw1.t();
        cv::Mat tcw1 = mpCurrentKeyFrame->GetTranslation();
        cv::Mat Tcw1(3, 4, CV_32F);
        Rcw1.copyTo(Tcw1.colRange(0, 3));
        tcw1.copyTo(Tcw1.col(3));

        // 得到当前关键帧在世界坐标系的坐标
        cv::Mat Ow1 = mpCurrentKeyFrame->GetCameraCenter();
        const float &fx1 = mpCurrentKeyFrame->fx;
        const float &fy1 = mpCurrentKeyFrame->fy;
        const float &cx1 = mpCurrentKeyFrame->cx;
        const float &cy1 = mpCurrentKeyFrame->cy;
        const float &invfx1 = mpCurrentKeyFrame->invfx;
        const float &invfy1 = mpCurrentKeyFrame->invfy;

        const float ratioFactor = 1.5f * mpCurrentKeyFrame->mfScaleFactor;

        int nnew = 0;

        // 2. 遍历相邻关键帧vpNeighKFs
        for (size_t i = 0; i < vpNeighKFs.size(); i++)
        {
            if (i > 0 && CheckNewKeyFrames())
                return;

            KeyFrame *pKF2 = vpNeighKFs[i];

            // 邻接的关键帧在世界坐标系中的坐标
            cv::Mat Ow2 = pKF2->GetCameraCenter();
            // 基线向量，两个关键帧间的相机位移
            cv::Mat vBaseline = Ow2 - Ow1;
            // 基线长度
            const float baseline = cv::norm(vBaseline);

            // 3. 判断相机运动的基线足够长
            // 邻接关键帧的场景深度中值
            const float medianDepthKF2 = pKF2->ComputeSceneMedianDepth(2);
            // baseline与景深的比例
            const float ratioBaselineDepth = baseline / medianDepthKF2;
            // 如果特别远(比例特别小)，那么不考虑当前邻接的关键帧，不生成3D点
            if (ratioBaselineDepth < 0.01)
                continue;

            // 4. 根据两个关键帧的位姿计算它们之间的基本矩阵
            cv::Mat F12 = ComputeF12(mpCurrentKeyFrame, pKF2);

            // 5. 通过极限约束限制匹配时的搜索范围，进行特征点匹配
            vector<pair<size_t, size_t>> vMatchedIndices;
            matcher.SearchForTriangulation(mpCurrentKeyFrame, pKF2, F12, vMatchedIndices);

            cv::Mat Rcw2 = pKF2->GetRotation();
            cv::Mat Rwc2 = Rcw2.t();
            cv::Mat tcw2 = pKF2->GetTranslation();
            cv::Mat Tcw2(3, 4, CV_32F);
            Rcw2.copyTo(Tcw2.colRange(0, 3));
            tcw2.copyTo(Tcw2.col(3));

            const float &fx2 = pKF2->fx;
            const float &fy2 = pKF2->fy;
            const float &cx2 = pKF2->cx;
            const float &cy2 = pKF2->cy;
            const float &invfx2 = pKF2->invfx;
            const float &invfy2 = pKF2->invfy;

            // 6. 对每对匹配通过三角化生成3D点， 和 triangulate函数差不多
            const int nmatches = vMatchedIndices.size();
            for (int ikp = 0; ikp < nmatches; ikp++)
            {
                // 当前匹配对在当前关键帧中的索引
                const int &idx1 = vMatchedIndices[ikp].first;

                // 当前匹配对在邻接关键帧中的索引
                const int &idx2 = vMatchedIndices[ikp].second;

                // 当前匹配在当前关键帧中的特征点
                const cv::KeyPoint &kp1 = mpCurrentKeyFrame->mvKeysUn[idx1];

                // 当前匹配在邻接关键帧中的特征点
                const cv::KeyPoint &kp2 = pKF2->mvKeysUn[idx2];

                // 利用匹配点反投影得到视差角
                // 特征点反投影
                cv::Mat xn1 = (cv::Mat_<float>(3, 1) << (kp1.pt.x - cx1) * invfx1, (kp1.pt.y - cy1) * invfy1, 1.0);
                cv::Mat xn2 = (cv::Mat_<float>(3, 1) << (kp2.pt.x - cx2) * invfx2, (kp2.pt.y - cy2) * invfy2, 1.0);

                // 由相机坐标系转到世界坐标系，得到视差角余弦值
                cv::Mat ray1 = Rwc1 * xn1;
                cv::Mat ray2 = Rwc2 * xn2;
                const float cosParallaxRays = ray1.dot(ray2) / (cv::norm(ray1) * cv::norm(ray2));

                // 三角化恢复3D点
                cv::Mat x3D;
                if (cosParallaxRays > 0 && cosParallaxRays < 0.9998)
                {
                    cv::Mat A(4, 4, CV_32F);
                    A.row(0) = xn1.at<float>(0) * Tcw1.row(2) - Tcw1.row(0);
                    A.row(1) = xn1.at<float>(1) * Tcw1.row(2) - Tcw1.row(1);
                    A.row(2) = xn2.at<float>(0) * Tcw2.row(2) - Tcw2.row(0);
                    A.row(3) = xn2.at<float>(1) * Tcw2.row(2) - Tcw2.row(1);

                    cv::Mat w, u, vt;
                    cv::SVD::compute(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

                    x3D = vt.row(3).t();

                    if (x3D.at<float>(3) == 0)
                        continue;

                    // Euclidean coordinates
                    x3D = x3D.rowRange(0, 3) / x3D.at<float>(3);
                }
                else
                    continue;

                cv::Mat x3Dt = x3D.t();

                // 步骤6.5：检测生成的3D点是否在相机前方
                float z1 = Rcw1.row(2).dot(x3Dt) + tcw1.at<float>(2);
                if (z1 <= 0)
                    continue;

                float z2 = Rcw2.row(2).dot(x3Dt) + tcw2.at<float>(2);
                if (z2 <= 0)
                    continue;

                //  步骤6.6：计算3D点在当前关键帧下的重投影误差
                const float &sigmaSquare1 = mpCurrentKeyFrame->mvLevelSigma2[kp1.octave];
                const float x1 = Rcw1.row(0).dot(x3Dt) + tcw1.at<float>(0);
                const float y1 = Rcw1.row(1).dot(x3Dt) + tcw1.at<float>(1);
                const float invz1 = 1.0 / z1;

                float u1 = fx1 * x1 * invz1 + cx1;
                float v1 = fy1 * y1 * invz1 + cy1;
                float errX1 = u1 - kp1.pt.x;
                float errY1 = v1 - kp1.pt.y;
                // 基于卡方检验计算出的阈值（假设测量有一个像素的偏差）
                if ((errX1 * errX1 + errY1 * errY1) > 5.991 * sigmaSquare1)
                    continue;

                // 计算3D点在另一个关键帧下的重投影误差
                const float sigmaSquare2 = pKF2->mvLevelSigma2[kp2.octave];
                const float x2 = Rcw2.row(0).dot(x3Dt) + tcw2.at<float>(0);
                const float y2 = Rcw2.row(1).dot(x3Dt) + tcw2.at<float>(1);
                const float invz2 = 1.0 / z2;

                float u2 = fx2 * x2 * invz2 + cx2;
                float v2 = fy2 * y2 * invz2 + cy2;
                float errX2 = u2 - kp2.pt.x;
                float errY2 = v2 - kp2.pt.y;
                if ((errX2 * errX2 + errY2 * errY2) > 5.991 * sigmaSquare2)
                    continue;

                // 步骤6.7：检查尺度连续性

                // 世界坐标系下，3D点与相机间的向量，方向由相机指向3D点
                cv::Mat normal1 = x3D - Ow1;
                float dist1 = cv::norm(normal1);

                cv::Mat normal2 = x3D - Ow2;
                float dist2 = cv::norm(normal2);

                if (dist1 == 0 || dist2 == 0)
                    continue;

                // ratioDist是不考虑金字塔尺度下的距离比例
                const float ratioDist = dist2 / dist1;
                // 金字塔尺度因子的比例
                const float ratioOctave = mpCurrentKeyFrame->mvScaleFactors[kp1.octave] / pKF2->mvScaleFactors[kp2.octave];

                /*if(fabs(ratioDist-ratioOctave)>ratioFactor)
                    continue;*/
                // ratioDist*ratioFactor < ratioOctave 或 ratioDist/ratioOctave > ratioFactor表明尺度变化是连续的
                if (ratioDist * ratioFactor < ratioOctave || ratioDist > ratioOctave * ratioFactor)
                    continue;

                // Triangulation is succesfull
                // 步骤6.8：三角化生成3D点成功，构造成MapPoint
                MapPoint *pMP = new MapPoint(x3D, mpCurrentKeyFrame, mpMap);

                // 步骤6.9：为该MapPoint添加属性：
                // a.观测到该MapPoint的关键帧
                // b.该MapPoint的描述子
                // c.该MapPoint的平均观测方向和深度范围
                pMP->AddObservation(mpCurrentKeyFrame, idx1);
                pMP->AddObservation(pKF2, idx2);

                mpCurrentKeyFrame->AddMapPoint(pMP, idx1);
                pKF2->AddMapPoint(pMP, idx2);

                pMP->ComputeDistinctiveDescriptors();

                pMP->UpdateNormalAndDepth();

                mpMap->AddMapPoint(pMP);

                // 步骤6.8：将新产生的点放入检测队列
                // 这些MapPoints都会经过MapPointCulling函数的检验
                mlpRecentAddedMapPoints.push_back(pMP);

                nnew++;
            }
        }
    }

    /**
     * 根据两关键帧的姿态计算两个关键帧之间的基本矩阵
     * @param  pKF1 关键帧1
     * @param  pKF2 关键帧2
     * @return      基本矩阵
     */
    cv::Mat LocalMapping::ComputeF12(KeyFrame *&pKF1, KeyFrame *&pKF2)
    {
        // Essential Matrix: t12叉乘R12
        // Fundamental Matrix: inv(K1)*E*inv(K2)

        cv::Mat R1w = pKF1->GetRotation();
        cv::Mat t1w = pKF1->GetTranslation();
        cv::Mat R2w = pKF2->GetRotation();
        cv::Mat t2w = pKF2->GetTranslation();

        cv::Mat R12 = R1w * R2w.t();
        cv::Mat t12 = -R1w * R2w.t() * t2w + t1w;

        cv::Mat t12x = SkewSymmetricMatrix(t12);

        const cv::Mat &K1 = pKF1->mK;
        const cv::Mat &K2 = pKF2->mK;

        return K1.t().inv() * t12x * R12 * K2.inv();
    }

    cv::Mat LocalMapping::SkewSymmetricMatrix(const cv::Mat &v)
    {
        return (cv::Mat_<float>(3, 3) << 0, -v.at<float>(2), v.at<float>(1),
                v.at<float>(2), 0, -v.at<float>(0),
                -v.at<float>(1), v.at<float>(0), 0);
    }

    /**
     * 检查并融合当前关键帧与相邻帧（两级相邻）重复的MapPoints
     */
    void LocalMapping::SearchInNeighbors()
    {
        // 1. 获得当前关键帧在covisibility图中权重排名前nn的邻接关键帧
        // 找到当前帧一级相邻与二级相邻关键帧
        int nn = 20;
        const vector<KeyFrame *> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);

        vector<KeyFrame *> vpTargetKFs;
        for (vector<KeyFrame *>::const_iterator vit = vpNeighKFs.begin(), vend = vpNeighKFs.end(); vit != vend; vit++)
        {
            KeyFrame *pKFi = *vit;
            if (pKFi->isBad() || pKFi->mnFuseTargetForKF == mpCurrentKeyFrame->mnID)
                continue;
            vpTargetKFs.push_back(pKFi);                       // 加入一级相邻帧
            pKFi->mnFuseTargetForKF = mpCurrentKeyFrame->mnID; // 并标记已经加入

            // Extend to some second neighbors
            const vector<KeyFrame *> vpSecondNeighKFs = pKFi->GetBestCovisibilityKeyFrames(5);
            for (vector<KeyFrame *>::const_iterator vit2 = vpSecondNeighKFs.begin(), vend2 = vpSecondNeighKFs.end(); vit2 != vend2; vit2++)
            {
                KeyFrame *pKFi2 = *vit2;
                if (pKFi2->isBad() || pKFi2->mnFuseTargetForKF == mpCurrentKeyFrame->mnID || pKFi2->mnID == mpCurrentKeyFrame->mnID)
                    continue;
                vpTargetKFs.push_back(pKFi2); // 存入二级相邻帧
            }
        }

        ORBmatcher matcher;

        // 2：将当前帧的MapPoints分别与一级二级相邻帧(的MapPoints)进行融合
        vector<MapPoint *> vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
        for (vector<KeyFrame *>::iterator vit = vpTargetKFs.begin(), vend = vpTargetKFs.end(); vit != vend; vit++)
        {
            KeyFrame *pKFi = *vit;

            // 投影当前帧的MapPoints到相邻关键帧pKFi中，并判断是否有重复的MapPoints
            // 1.如果MapPoint能匹配关键帧的特征点，并且该点有对应的MapPoint，那么将两个MapPoint合并（选择观测数多的）
            // 2.如果MapPoint能匹配关键帧的特征点，并且该点没有对应的MapPoint，那么为该点添加MapPoint
            matcher.Fuse(pKFi, vpMapPointMatches);
        }

        // 用于存储一级邻接和二级邻接关键帧所有MapPoints的集合
        vector<MapPoint *> vpFuseCandidates;
        vpFuseCandidates.reserve(vpTargetKFs.size() * vpMapPointMatches.size());

        // 3：将一级二级相邻帧的MapPoints分别与当前帧（的MapPoints）进行融合
        // 遍历每一个一级邻接和二级邻接关键帧
        for (vector<KeyFrame *>::iterator vitKF = vpTargetKFs.begin(), vendKF = vpTargetKFs.end(); vitKF != vendKF; vitKF++)
        {
            KeyFrame *pKFi = *vitKF;

            vector<MapPoint *> vpMapPointsKFi = pKFi->GetMapPointMatches();

            // 遍历当前一级邻接和二级邻接关键帧中所有的MapPoints
            for (vector<MapPoint *>::iterator vitMP = vpMapPointsKFi.begin(), vendMP = vpMapPointsKFi.end(); vitMP != vendMP; vitMP++)
            {
                MapPoint *pMP = *vitMP;
                if (!pMP)
                    continue;

                // 判断MapPoints是否为坏点，或者是否已经加进集合vpFuseCandidates
                if (pMP->isBad() || pMP->mnFuseCandidateForKF == mpCurrentKeyFrame->mnID)
                    continue;

                // 加入集合，并标记已经加入
                pMP->mnFuseCandidateForKF = mpCurrentKeyFrame->mnID;
                vpFuseCandidates.push_back(pMP);
            }
        }

        matcher.Fuse(mpCurrentKeyFrame, vpFuseCandidates);

        // 步骤4：更新当前帧MapPoints的描述子，深度，观测主方向等属性
        vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
        for (size_t i = 0, iend = vpMapPointMatches.size(); i < iend; i++)
        {
            MapPoint *pMP = vpMapPointMatches[i];
            if (pMP)
            {
                if (!pMP->isBad())
                {
                    // 在所有找到pMP的关键帧中，获得最佳的描述子
                    pMP->ComputeDistinctiveDescriptors();

                    // 更新平均观测方向和观测距离
                    pMP->UpdateNormalAndDepth();
                }
            }
        }

        // 步骤5：更新当前帧的MapPoints后更新与其它帧的连接关系
        // 更新covisibility图
        mpCurrentKeyFrame->UpdateConnections();
    }

}