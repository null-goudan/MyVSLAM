#include "ORBmatcher.h"
#include <limits.h>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <stdint.h>

using namespace std;

namespace Goudan_SLAM
{
    const int ORBmatcher::TH_HIGH = 100;
    const int ORBmatcher::TH_LOW = 50;
    const int ORBmatcher::HISTO_LENGTH = 30;

    ORBmatcher::ORBmatcher(float nnratio, bool checkOri)
        : mfNNratio(nnratio), mbCheckOrientation(checkOri) {}

    int ORBmatcher::SearchForInitialization(Frame &F1, Frame &F2, std::vector<cv::Point2f> &vbPrevMatched, std::vector<int> &vnMatches12, int windowSize)
    {
        int nmatches = 0;
        vnMatches12 = vector<int>(F1.mvKeysUn.size(), -1);

        vector<int> rotHist[HISTO_LENGTH];
        for (int i = 0; i < HISTO_LENGTH; i++)
            rotHist[i].reserve(500);
        const float factor = HISTO_LENGTH / 360.0f;

        vector<int> vMatchedDistance(F2.mvKeysUn.size(), INT_MAX);
        vector<int> vnMatches21(F2.mvKeysUn.size(), -1);

        for (size_t i1 = 0, iend1 = F1.mvKeysUn.size(); i1 < iend1; i1++)
        {
            cv::KeyPoint kp1 = F1.mvKeysUn[i1];
            int level1 = kp1.octave;
            if (level1 > 0)
                continue;

            vector<size_t> vIndices2 = F2.GetFeaturesInArea(vbPrevMatched[i1].x, vbPrevMatched[i1].y, windowSize, level1, level1);

            if (vIndices2.empty())
                continue;

            cv::Mat d1 = F1.mDescriptors.row(i1);

            int bestDist = INT_MAX;
            int bestDist2 = INT_MAX;
            int bestIdx2 = -1;

            for (vector<size_t>::iterator vit = vIndices2.begin(); vit != vIndices2.end(); vit++)
            {
                size_t i2 = *vit;

                cv::Mat d2 = F2.mDescriptors.row(i2);

                int dist = DescriptorDistance(d1, d2);

                if (vMatchedDistance[i2] <= dist)
                    continue;

                if (dist < bestDist)
                {
                    bestDist2 = bestDist;
                    bestDist = dist;
                    bestIdx2 = i2;
                }
                else if (dist < bestDist2)
                {
                    bestDist2 = dist;
                }
            }
            // 根据阈值和角度投票剔除误匹配
            if (bestDist <= TH_LOW)
            {
                // 最佳匹配比次佳匹配要好很多才能可靠
                if (bestDist < (float)bestDist2 * mfNNratio)
                {
                    if (vnMatches21[bestIdx2] >= 0)
                    {
                        vnMatches12[vnMatches21[bestIdx2]] = -1;
                        nmatches--;
                    }
                    vnMatches12[i1] = bestIdx2;
                    vnMatches21[bestIdx2] = i1;
                    vMatchedDistance[bestDist2] = bestDist;
                    nmatches++;

                    if (mbCheckOrientation)
                    {
                        // trick!
                        // angle：每个特征点在提取描述子时的旋转主方向角度，如果图像旋转了，这个角度将发生改变
                        // 所有的特征点的角度变化应该是一致的，通过直方图统计得到最准确的角度变化值
                        float rot = F1.mvKeysUn[i1].angle - F2.mvKeysUn[bestIdx2].angle;
                        if (rot < 0.0)
                            rot += 360.0f;
                        int bin = round(rot * factor);
                        if (bin == HISTO_LENGTH)
                            bin = 0;
                        assert(bin >= 0 && bin < HISTO_LENGTH);
                        rotHist[bin].push_back(i1);
                    }
                }
            }
        }
        // 剔除不佳匹配
        if (mbCheckOrientation)
        {
            int ind1 = -1;
            int ind2 = -1;
            int ind3 = -1;

            ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3);
            for (int i = 0; i < HISTO_LENGTH; i++)
            {
                if (i == ind1 || i == ind2 || i == ind3)
                    continue;
                for (size_t j = 0, jend = rotHist[i].size(); j < jend; j++)
                {
                    int idx1 = rotHist[i][j];
                    if (vnMatches12[idx1] >= 0)
                    {
                        vnMatches12[idx1] = -1;
                        nmatches--;
                    }
                }
            }
        }

        // Update prev matched
        for (size_t i1 = 0, iend1 = vnMatches12.size(); i1 < iend1; i1++)
            if (vnMatches12[i1] >= 0)
                vbPrevMatched[i1] = F2.mvKeysUn[vnMatches12[i1]].pt;

        return nmatches;
    }

    int ORBmatcher::SearchByBoW(KeyFrame *pKF, Frame &F, vector<MapPoint *> &vpMapPointMatches)
    {
        cout << "Start SearchByBow.." << endl;

        const vector<MapPoint *> vpMapPointsKF = pKF->GetMapPointMatches();
        cout << "Reference KeyFrame MapPoint Num: " << vpMapPointsKF.size() << endl;
        vpMapPointMatches = vector<MapPoint *>(F.N, static_cast<MapPoint *>(NULL));

        const DBoW2::FeatureVector &vFeatVecKF = pKF->mFeatVec;

        int nmatches = 0;

        vector<int> rotHist[HISTO_LENGTH];
        for (int i = 0; i < HISTO_LENGTH; i++)
            rotHist[i].reserve(500);

        const float factor = HISTO_LENGTH / 360.0f;

        // 将属于同一节点（特定层）的ORB特征进行匹配
        DBoW2::FeatureVector::const_iterator KFit = vFeatVecKF.begin();
        DBoW2::FeatureVector::const_iterator Fit = F.mFeatVec.begin();
        DBoW2::FeatureVector::const_iterator KFend = vFeatVecKF.end();
        DBoW2::FeatureVector::const_iterator Fend = F.mFeatVec.end();

        while (KFit != KFend && Fit != Fend) // 使用二分查找(利用BoW的特点, 迅速找到其匹配的特征值)
        {
            if (KFit->first == Fit->first) // 取出属于同一个node的ORB特征点
            {
                const vector<unsigned int> vIndicesKF = KFit->second;
                const vector<unsigned int> vIndicesF = Fit->second;

                // 遍历KF中的属于该node的特征点
                for (size_t iKF = 0; iKF < vIndicesKF.size(); iKF++)
                {
                    const unsigned int realIdxKF = vIndicesKF[iKF];

                    MapPoint *pMP = vpMapPointsKF[realIdxKF]; // 取出KF中该特征对应的MapPoint

                    if (!pMP)
                        continue;

                    if (pMP->isBad())
                        continue;

                    const cv::Mat &dKF = pKF->mDescriptors.row(realIdxKF);

                    int bestDist1 = 256; // 最好的距离 最小距离
                    int bestIdxF = -1;
                    int bestDist2 = 256; // 倒数第二好距离

                    // 遍历F中属于该node的特征点， 找到最佳匹配点
                    for (size_t iF = 0; iF < vIndicesF.size(); iF++)
                    {
                        const unsigned int realIdxF = vIndicesF[iF];

                        if (vpMapPointMatches[realIdxF]) // 表明这个点已经被匹配过了，不再匹配，加快速度
                            continue;

                        const cv::Mat &dF = F.mDescriptors.row(realIdxF); // 取出F中该特征对应的描述子

                        const int dist = DescriptorDistance(dKF, dF); // 求描述子的距离

                        if (dist < bestDist1) // dist < bestDist1 < bestDist2，更新bestDist1 bestDist2
                        {
                            bestDist2 = bestDist1;
                            bestDist1 = dist;
                            bestIdxF = realIdxF;
                        }
                        else if (dist < bestDist2) // bestDist1 < dist < bestDist2，更新bestDist2
                        {
                            bestDist2 = dist;
                        }
                    } // for

                    // 根据阈值 和 角度投票剔除误匹配
                    if (bestDist1 <= TH_LOW) // 匹配距离（误差）小于阈值
                    {
                        // trick!
                        // 最佳匹配比次佳匹配明显要好，那么最佳匹配才真正靠谱
                        if (static_cast<float>(bestDist1) < mfNNratio * static_cast<float>(bestDist2))
                        {
                            // 步骤5：更新特征点的MapPoint
                            vpMapPointMatches[bestIdxF] = pMP;

                            const cv::KeyPoint &kp = pKF->mvKeysUn[realIdxKF];

                            if (mbCheckOrientation)
                            {
                                // trick!
                                // angle：每个特征点在提取描述子时的旋转主方向角度，如果图像旋转了，这个角度将发生改变
                                // 所有的特征点的角度变化应该是一致的，通过直方图统计得到最准确的角度变化值
                                float rot = kp.angle - F.mvKeys[bestIdxF].angle; // 该特征点的角度变化值
                                if (rot < 0.0)
                                    rot += 360.0f;
                                int bin = round(rot * factor); // 将rot分配到bin组
                                if (bin == HISTO_LENGTH)
                                    bin = 0;
                                assert(bin >= 0 && bin < HISTO_LENGTH);
                                rotHist[bin].push_back(bestIdxF);
                            }
                            nmatches++;
                        }
                    }
                }

                KFit++;
                Fit++;
            }
            else if (KFit->first < Fit->first)
            {
                KFit = vFeatVecKF.lower_bound(Fit->first);
            }
            else
            {
                Fit = vFeatVecKF.lower_bound(KFit->first);
            }
        } // while

        // 根据方向剔除误匹配的点
        if (mbCheckOrientation)
        {
            int ind1 = -1;
            int ind2 = -1;
            int ind3 = -1;

            // 计算rotHist中最大的三个的index
            ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3);

            for (int i = 0; i < HISTO_LENGTH; i++)
            {
                // 如果特征点的旋转角度变化量属于这三个组，则保留
                if (i == ind1 || i == ind2 || i == ind3)
                    continue;

                // 将除了ind1 ind2 ind3以外的匹配点去掉
                for (size_t j = 0, jend = rotHist[i].size(); j < jend; j++)
                {
                    vpMapPointMatches[rotHist[i][j]] = static_cast<MapPoint *>(NULL);
                    nmatches--;
                }
            }
        }
        cout << "End SearchByBow: nmathes: " << nmatches << endl;
        return nmatches;
    }

    /**
     * @brief 对上一帧每个3D点通过投影在小范围内找到和最匹配的2D点。从而实现当前帧CurrentFrame对上一帧LastFrame 3D点的匹配跟踪。用于tracking中前后帧跟踪
     *
     * 上一帧中包含了MapPoints，对这些MapPoints进行tracking，由此增加当前帧的MapPoints \n
     * 1. 将上一帧的MapPoints投影到当前帧(根据速度模型可以估计当前帧的Tcw)
     * 2. 在投影点附近根据描述子距离选取匹配，以及最终的方向投票机制进行剔除
     * @param  CurrentFrame 当前帧
     * @param  LastFrame    上一帧
     * @param  th           阈值
     * @return              成功匹配的数量
     * @see SearchByBoW()
     */
    int ORBmatcher::SearchByProjection(Frame &CurrentFrame, const Frame &LastFrame, const float th)
    {
        int nmatches = 0;

        // Rotation Histogram (to check rotation consistency)
        vector<int> rotHist[HISTO_LENGTH];
        for (int i = 0; i < HISTO_LENGTH; i++)
            rotHist[i].reserve(500);
        const float factor = HISTO_LENGTH / 360.0f;

        const cv::Mat Rcw = CurrentFrame.mTcw.rowRange(0, 3).colRange(0, 3);
        const cv::Mat tcw = CurrentFrame.mTcw.rowRange(0, 3).col(3);

        const cv::Mat twc = -Rcw.t() * tcw; // twc(w)

        const cv::Mat Rlw = LastFrame.mTcw.rowRange(0, 3).colRange(0, 3);
        const cv::Mat tlw = LastFrame.mTcw.rowRange(0, 3).col(3); // tlw(l)

        // vector from LastFrame to CurrentFrame expressed in LastFrame
        const cv::Mat tlc = Rlw * twc + tlw; // Rlw*twc(w) = twc(l), twc(l) + tlw(l) = tlc(l)

        for (int i = 0; i < LastFrame.N; i++)
        {
            MapPoint *pMP = LastFrame.mvpMapPoints[i];

            if (pMP)
            {
                if (!LastFrame.mvbOutlier[i])
                {
                    // 对上一帧有效的MapPoints进行跟踪
                    // Project
                    cv::Mat x3Dw = pMP->GetWorldPos();
                    cv::Mat x3Dc = Rcw * x3Dw + tcw;

                    const float xc = x3Dc.at<float>(0);
                    const float yc = x3Dc.at<float>(1);
                    const float invzc = 1.0 / x3Dc.at<float>(2);

                    if (invzc < 0)
                        continue;

                    float u = CurrentFrame.fx * xc * invzc + CurrentFrame.cx;
                    float v = CurrentFrame.fy * yc * invzc + CurrentFrame.cy;

                    if (u < CurrentFrame.mnMinX || u > CurrentFrame.mnMaxX)
                        continue;
                    if (v < CurrentFrame.mnMinY || v > CurrentFrame.mnMaxY)
                        continue;

                    int nLastOctave = LastFrame.mvKeys[i].octave;

                    // Search in a window. Size depends on scale
                    float radius = th * CurrentFrame.mvScaleFactors[nLastOctave]; // 尺度越大，搜索范围越大

                    vector<size_t> vIndices2;

                    // NOTE 尺度越大,图像越小
                    // 以下可以这么理解，例如一个有一定面积的圆点，在某个尺度n下它是一个特征点
                    // 当前进时，圆点的面积增大，在某个尺度m下它是一个特征点，由于面积增大，则需要在更高的尺度下才能检测出来
                    vIndices2 = CurrentFrame.GetFeaturesInArea(u, v, radius, nLastOctave - 1, nLastOctave + 1);

                    if (vIndices2.empty())
                        continue;

                    const cv::Mat dMP = pMP->GetDescriptor();

                    int bestDist = 256;
                    int bestIdx2 = -1;

                    // 遍历满足条件的特征点
                    for (vector<size_t>::const_iterator vit = vIndices2.begin(), vend = vIndices2.end(); vit != vend; vit++)
                    {
                        // 如果该特征点已经有对应的MapPoint了,则退出该次循环
                        const size_t i2 = *vit;
                        if (CurrentFrame.mvpMapPoints[i2])
                            if (CurrentFrame.mvpMapPoints[i2]->Observations() > 0)
                                continue;

                        const cv::Mat &d = CurrentFrame.mDescriptors.row(i2);

                        const int dist = DescriptorDistance(dMP, d);

                        if (dist < bestDist)
                        {
                            bestDist = dist;
                            bestIdx2 = i2;
                        }
                    }

                    // 详见SearchByBoW(KeyFrame* pKF,Frame &F, vector<MapPoint*> &vpMapPointMatches)函数步骤4
                    if (bestDist <= TH_HIGH)
                    {
                        CurrentFrame.mvpMapPoints[bestIdx2] = pMP; // 为当前帧添加MapPoint
                        nmatches++;

                        if (mbCheckOrientation)
                        {
                            float rot = LastFrame.mvKeysUn[i].angle - CurrentFrame.mvKeysUn[bestIdx2].angle;
                            if (rot < 0.0)
                                rot += 360.0f;
                            int bin = round(rot * factor);
                            if (bin == HISTO_LENGTH)
                                bin = 0;
                            assert(bin >= 0 && bin < HISTO_LENGTH);
                            rotHist[bin].push_back(bestIdx2);
                        }
                    }
                }
            }
        }

        // Apply rotation consistency
        if (mbCheckOrientation)
        {
            int ind1 = -1;
            int ind2 = -1;
            int ind3 = -1;

            ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3);

            for (int i = 0; i < HISTO_LENGTH; i++)
            {
                if (i != ind1 && i != ind2 && i != ind3)
                {
                    for (size_t j = 0, jend = rotHist[i].size(); j < jend; j++)
                    {
                        CurrentFrame.mvpMapPoints[rotHist[i][j]] = static_cast<MapPoint *>(NULL);
                        nmatches--;
                    }
                }
            }
        }

        return nmatches;
    }

    /**
     * @brief 对于每个局部3D点通过投影在小范围内找到和最匹配的2D点。从而实现Frame对Local MapPoint的跟踪。用于tracking过程中实现当前帧对局部3D点的跟踪。
     *
     * 将Local MapPoint投影到当前帧中, 由此增加当前帧的MapPoints \n
     * 在SearchLocalPoints()中已经将Local MapPoints重投影（isInFrustum()）到当前帧 \n
     * 并标记了这些点是否在当前帧的视野中，即mbTrackInView \n
     * 对这些MapPoints，在其投影点附近根据描述子距离选取匹配，以及最终的方向投票机制进行剔除
     * @param  F           当前帧
     * @param  vpMapPoints Local MapPoints
     * @param  th          搜索范围因子：r = r * th * ScaleFactor
     * @return             成功匹配的数量
     * @see SearchLocalPoints() isInFrustum()
     */
    int ORBmatcher::SearchByProjection(Frame &F, const vector<MapPoint *> &vpMapPoints, const float th)
    {
        int nmatches = 0;
        const bool bFactor = th != 1.0;

        for (size_t iMP = 0; iMP < vpMapPoints.size(); iMP++)
        {
            MapPoint *pMP = vpMapPoints[iMP];
            // 判断该点是否投影
            if (!pMP->mbTrackInView)
                continue;

            if (pMP->isBad())
                continue;

            // 1. 通过举例预测特征点所在的金字塔层数
            const int &nPredictedLevel = pMP->mnTrackScaleLevel;

            // 2. 根据观测到该3D点的视角确定搜索窗口的大小， 若相机正对着该点则r取一个较小的值
            float r = RadiusByViewingCos(pMP->mTrackViewCos);

            if (bFactor)
                r *= th;

            // 3. 在2D投影点附近一定范围内搜索属于miniLevel~maxLevel层的特征点
            const vector<size_t> vIndices =
                F.GetFeaturesInArea(pMP->mTrackProjX, pMP->mTrackProjY, r * F.mvScaleFactors[nPredictedLevel], nPredictedLevel - 1, nPredictedLevel);

            if (vIndices.empty())
                continue;

            const cv::Mat MPdescriptor = pMP->GetDescriptor();

            int bestDist = 256;
            int bestLevel = -1;
            int bestDist2 = 256;
            int bestLevel2 = -1;
            int bestIdx = -1;

            // Get best and second matches with near keypoints
            // step4：在vIndices内找到最佳匹配与次佳匹配，如果最优匹配误差小于阈值，且最优匹配明显优于次优匹配，则匹配3D点-2D特征点匹配关联成功
            for (vector<size_t>::const_iterator vit = vIndices.begin(), vend = vIndices.end(); vit != vend; vit++)
            {
                const size_t idx = *vit;

                // 如果Frame中的该兴趣点已经有对应的MapPoint了，则退出该次循环
                if (F.mvpMapPoints[idx])
                    if (F.mvpMapPoints[idx]->Observations() > 0)
                        continue;

                const cv::Mat &d = F.mDescriptors.row(idx);

                const int dist = DescriptorDistance(MPdescriptor, d);

                // 记录最优匹配和次优匹配
                if (dist < bestDist)
                {
                    bestDist2 = bestDist;
                    bestDist = dist;
                    bestLevel2 = bestLevel;
                    bestLevel = F.mvKeysUn[idx].octave;
                    bestIdx = idx;
                }
                else if (dist < bestDist2)
                {
                    bestLevel2 = F.mvKeysUn[idx].octave;
                    bestDist2 = dist;
                }
            }

            // Apply ratio to second match (only if best and second are in the same scale level)
            if (bestDist <= TH_HIGH)
            {
                if (bestLevel == bestLevel2 && bestDist > mfNNratio * bestDist2)
                    continue;

                F.mvpMapPoints[bestIdx] = pMP; // 为Frame中的兴趣点增加对应的MapPoint
                nmatches++;
            }
        }

        return nmatches;
    }

    int ORBmatcher::SearchForTriangulation(KeyFrame *pKF1, KeyFrame *pKF2, cv::Mat F12,
                                           std::vector<pair<size_t, size_t>> &vMatchedPairs)
    {
        const DBoW2::FeatureVector &vFeatVec1 = pKF1->mFeatVec;
        const DBoW2::FeatureVector &vFeatVec2 = pKF2->mFeatVec;

        // Compute epipole in second image
        // 计算KF1的相机中心在KF2图像平面的坐标，即极点坐标
        cv::Mat Cw = pKF1->GetCameraCenter(); // tw2c1
        cv::Mat R2w = pKF2->GetRotation();    // Rc2w
        cv::Mat t2w = pKF2->GetTranslation(); // tc2w
        cv::Mat C2 = R2w * Cw + t2w;          // tc2c1 KF1的相机中心在KF2坐标系的表示
        const float invz = 1.0f / C2.at<float>(2);
        // 步骤0：得到KF1的相机光心在KF2中的坐标（KF1在KF2中的极点坐标）
        const float ex = pKF2->fx * C2.at<float>(0) * invz + pKF2->cx;
        const float ey = pKF2->fy * C2.at<float>(1) * invz + pKF2->cy;

        // Find matches between not tracked keypoints
        // Matching speed-up by ORB Vocabulary
        // Compare only ORB that share the same node

        int nmatches = 0;
        vector<bool> vbMatched2(pKF2->N, false);
        vector<int> vMatches12(pKF1->N, -1);

        vector<int> rotHist[HISTO_LENGTH];
        for (int i = 0; i < HISTO_LENGTH; i++)
            rotHist[i].reserve(500);

        const float factor = HISTO_LENGTH / 360.0f;

        // We perform the matching over ORB that belong to the same vocabulary node (at a certain level)
        // 将属于同一节点(特定层)的ORB特征进行匹配
        // FeatureVector的数据结构类似于：{(node1,feature_vector1) (node2,feature_vector2)...}
        // f1it->first对应node编号，f1it->second对应属于该node的所有特特征点编号
        DBoW2::FeatureVector::const_iterator f1it = vFeatVec1.begin();
        DBoW2::FeatureVector::const_iterator f2it = vFeatVec2.begin();
        DBoW2::FeatureVector::const_iterator f1end = vFeatVec1.end();
        DBoW2::FeatureVector::const_iterator f2end = vFeatVec2.end();

        // 步骤1：遍历pKF1和pKF2中的node节点
        while (f1it != f1end && f2it != f2end)
        {
            // 如果f1it和f2it属于同一个node节点
            if (f1it->first == f2it->first)
            {
                // 步骤2：遍历该node节点下(f1it->first)的所有特征点
                for (size_t i1 = 0, iend1 = f1it->second.size(); i1 < iend1; i1++)
                {
                    // 获取pKF1中属于该node节点的所有特征点索引
                    const size_t idx1 = f1it->second[i1];

                    // 步骤2.1：通过特征点索引idx1在pKF1中取出对应的MapPoint
                    MapPoint *pMP1 = pKF1->GetMapPoint(idx1);

                    // If there is already a MapPoint skip
                    // ！！！！！！由于寻找的是未匹配的特征点，所以pMP1应该为NULL
                    if (pMP1)
                        continue;

                    // 步骤2.2：通过特征点索引idx1在pKF1中取出对应的特征点
                    const cv::KeyPoint &kp1 = pKF1->mvKeysUn[idx1];

                    // 步骤2.3：通过特征点索引idx1在pKF1中取出对应的特征点的描述子
                    const cv::Mat &d1 = pKF1->mDescriptors.row(idx1);

                    int bestDist = TH_LOW;
                    int bestIdx2 = -1;

                    // 步骤3：遍历该node节点下(f2it->first)的所有特征点
                    for (size_t i2 = 0, iend2 = f2it->second.size(); i2 < iend2; i2++)
                    {
                        // 获取pKF2中属于该node节点的所有特征点索引
                        size_t idx2 = f2it->second[i2];

                        // 步骤3.1：通过特征点索引idx2在pKF2中取出对应的MapPoint
                        MapPoint *pMP2 = pKF2->GetMapPoint(idx2);

                        // If we have already matched or there is a MapPoint skip
                        // 如果pKF2当前特征点索引idx2已经被匹配过或者对应的3d点非空
                        // 那么这个索引idx2就不能被考虑
                        if (vbMatched2[idx2] || pMP2)
                            continue;

                        // 步骤3.2：通过特征点索引idx2在pKF2中取出对应的特征点的描述子
                        const cv::Mat &d2 = pKF2->mDescriptors.row(idx2);

                        // 计算idx1与idx2在两个关键帧中对应特征点的描述子距离
                        const int dist = DescriptorDistance(d1, d2);

                        if (dist > TH_LOW || dist > bestDist)
                            continue;

                        // 步骤3.3：通过特征点索引idx2在pKF2中取出对应的特征点
                        const cv::KeyPoint &kp2 = pKF2->mvKeysUn[idx2];

                        const float distex = ex - kp2.pt.x;
                        const float distey = ey - kp2.pt.y;
                        // ！！！！该特征点距离极点太近，表明kp2对应的MapPoint距离pKF1相机太近
                        if (distex * distex + distey * distey < 100 * pKF2->mvScaleFactors[kp2.octave])
                            continue;

                        // 步骤4：计算特征点kp2到kp1极线（kp1对应pKF2的一条极线）的距离是否小于阈值
                        if (CheckDistEpipolarLine(kp1, kp2, F12, pKF2))
                        {
                            bestIdx2 = idx2;
                            bestDist = dist;
                        }
                    }

                    // 步骤1、2、3、4总结下来就是：将左图像的每个特征点与右图像同一node节点的所有特征点
                    // 依次检测，判断是否满足对极几何约束，满足约束就是匹配的特征点

                    // 详见SearchByBoW(KeyFrame* pKF,Frame &F, vector<MapPoint*> &vpMapPointMatches)函数步骤4
                    if (bestIdx2 >= 0)
                    {
                        const cv::KeyPoint &kp2 = pKF2->mvKeysUn[bestIdx2];
                        vMatches12[idx1] = bestIdx2;
                        vbMatched2[bestIdx2] = true;
                        nmatches++;

                        if (mbCheckOrientation)
                        {
                            float rot = kp1.angle - kp2.angle;
                            if (rot < 0.0)
                                rot += 360.0f;
                            int bin = round(rot * factor);
                            if (bin == HISTO_LENGTH)
                                bin = 0;
                            assert(bin >= 0 && bin < HISTO_LENGTH);
                            rotHist[bin].push_back(idx1);
                        }
                    }
                }

                f1it++;
                f2it++;
            }
            else if (f1it->first < f2it->first)
            {
                f1it = vFeatVec1.lower_bound(f2it->first);
            }
            else
            {
                f2it = vFeatVec2.lower_bound(f1it->first);
            }
        }

        if (mbCheckOrientation)
        {
            int ind1 = -1;
            int ind2 = -1;
            int ind3 = -1;

            ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3);

            for (int i = 0; i < HISTO_LENGTH; i++)
            {
                if (i == ind1 || i == ind2 || i == ind3)
                    continue;
                for (size_t j = 0, jend = rotHist[i].size(); j < jend; j++)
                {
                    vMatches12[rotHist[i][j]] = -1;
                    nmatches--;
                }
            }
        }

        vMatchedPairs.clear();
        vMatchedPairs.reserve(nmatches);

        for (size_t i = 0, iend = vMatches12.size(); i < iend; i++)
        {
            if (vMatches12[i] < 0)
                continue;
            vMatchedPairs.push_back(make_pair(i, vMatches12[i]));
        }

        return nmatches;
    }

    bool ORBmatcher::CheckDistEpipolarLine(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const cv::Mat &F12, const KeyFrame *pKF2)
    {
        // Epipolar line in second image l = x1'F12 = [a b c]
        // 求出kp1在pKF2上对应的极线
        const float a = kp1.pt.x * F12.at<float>(0, 0) + kp1.pt.y * F12.at<float>(1, 0) + F12.at<float>(2, 0);
        const float b = kp1.pt.x * F12.at<float>(0, 1) + kp1.pt.y * F12.at<float>(1, 1) + F12.at<float>(2, 1);
        const float c = kp1.pt.x * F12.at<float>(0, 2) + kp1.pt.y * F12.at<float>(1, 2) + F12.at<float>(2, 2);

        // 计算kp2特征点到极线的距离：
        // 极线l：ax + by + c = 0
        // (u,v)到l的距离为： |au+bv+c| / sqrt(a^2+b^2)

        const float num = a * kp2.pt.x + b * kp2.pt.y + c;

        const float den = a * a + b * b;

        if (den == 0)
            return false;

        const float dsqr = num * num / den;

        // 尺度越大，范围应该越大。
        // 金字塔最底层一个像素就占一个像素，在倒数第二层，一个像素等于最底层1.2个像素（假设金字塔尺度为1.2）
        return dsqr < 3.84 * pKF2->mvLevelSigma2[kp2.octave];
    }

    /**
     * @brief 将MapPoints投影（用关键帧的位姿）到关键帧pKF中，并判断是否有重复的MapPoints
     * 1.如果MapPoint能匹配关键帧的特征点，并且该点有对应的MapPoint，那么将两个MapPoint合并（选择观测数多的）
     * 2.如果MapPoint能匹配关键帧的特征点，并且该点没有对应的MapPoint，那么为该点添加MapPoint
     * @param  pKF         相邻关键帧
     * @param  vpMapPoints 当前关键帧的MapPoints
     * @param  th          搜索半径的因子
     * @return             重复MapPoints的数量
     */
    int ORBmatcher::Fuse(KeyFrame *pKF, const vector<MapPoint *> &vpMapPoints, const float th)
    {
        cv::Mat Rcw = pKF->GetRotation();
        cv::Mat tcw = pKF->GetTranslation();

        const float &fx = pKF->fx;
        const float &fy = pKF->fy;
        const float &cx = pKF->cx;
        const float &cy = pKF->cy;
        const float &bf = pKF->mbf;

        cv::Mat Ow = pKF->GetCameraCenter();

        int nFused = 0;

        const int nMPs = vpMapPoints.size();

        // 遍历所有的MapPoints
        for (int i = 0; i < nMPs; i++)
        {
            MapPoint *pMP = vpMapPoints[i];

            if (!pMP)
                continue;

            if (pMP->isBad() || pMP->IsInKeyFrame(pKF))
                continue;

            cv::Mat p3Dw = pMP->GetWorldPos();
            cv::Mat p3Dc = Rcw * p3Dw + tcw;

            // Depth must be positive
            if (p3Dc.at<float>(2) < 0.0f)
                continue;

            const float invz = 1 / p3Dc.at<float>(2);
            const float x = p3Dc.at<float>(0) * invz;
            const float y = p3Dc.at<float>(1) * invz;

            const float u = fx * x + cx;
            const float v = fy * y + cy; // 步骤1：得到MapPoint在图像上的投影坐标

            // Point must be inside the image
            if (!pKF->IsInImage(u, v))
                continue;

            const float maxDistance = pMP->GetMaxDistanceInvariance();
            const float minDistance = pMP->GetMinDistanceInvariance();
            cv::Mat PO = p3Dw - Ow;
            const float dist3D = cv::norm(PO);

            // Depth must be inside the scale pyramid of the image
            if (dist3D < minDistance || dist3D > maxDistance)
                continue;

            // Viewing angle must be less than 60 deg
            cv::Mat Pn = pMP->GetNormal();

            if (PO.dot(Pn) < 0.5 * dist3D)
                continue;

            int nPredictedLevel = pMP->PredictScale(dist3D, pKF);

            // Search in a radius
            const float radius = th * pKF->mvScaleFactors[nPredictedLevel]; // 步骤2：根据MapPoint的深度确定尺度，从而确定搜索范围

            const vector<size_t> vIndices = pKF->GetFeaturesInArea(u, v, radius);

            if (vIndices.empty())
                continue;

            // Match to the most similar keypoint in the radius

            const cv::Mat dMP = pMP->GetDescriptor();

            int bestDist = 256;
            int bestIdx = -1;
            for (vector<size_t>::const_iterator vit = vIndices.begin(), vend = vIndices.end(); vit != vend; vit++) // 步骤3：遍历搜索范围内的features
            {
                const size_t idx = *vit;

                const cv::KeyPoint &kp = pKF->mvKeysUn[idx];

                const int &kpLevel = kp.octave;

                if (kpLevel < nPredictedLevel - 1 || kpLevel > nPredictedLevel)
                    continue;


                const float &kpx = kp.pt.x;
                const float &kpy = kp.pt.y;
                const float ex = u - kpx;
                const float ey = v - kpy;
                const float e2 = ex * ex + ey * ey;

                // 基于卡方检验计算出的阈值（假设测量有一个像素的偏差）
                if (e2 * pKF->mvInvLevelSigma2[kpLevel] > 5.99)
                    continue;

                const cv::Mat &dKF = pKF->mDescriptors.row(idx);

                const int dist = DescriptorDistance(dMP, dKF);

                if (dist < bestDist) // 找MapPoint在该区域最佳匹配的特征点
                {
                    bestDist = dist;
                    bestIdx = idx;
                }
            }

            // If there is already a MapPoint replace otherwise add new measurement
            if (bestDist <= TH_LOW) // 找到了MapPoint在该区域最佳匹配的特征点
            {
                MapPoint *pMPinKF = pKF->GetMapPoint(bestIdx);
                if (pMPinKF) // 如果这个点有对应的MapPoint
                {
                    if (!pMPinKF->isBad()) // 如果这个MapPoint不是bad，选择哪一个呢？
                    {
                        if (pMPinKF->Observations() > pMP->Observations())
                            pMP->Replace(pMPinKF);
                        else
                            pMPinKF->Replace(pMP);
                    }
                }
                else // 如果这个点没有对应的MapPoint
                {
                    pMP->AddObservation(pKF, bestIdx);
                    pKF->AddMapPoint(pMP, bestIdx);
                }
                nFused++;
            }
        }

        return nFused;
    }
    // 根据观测角度决定 SearchByProjection 的搜索范围
    float ORBmatcher::RadiusByViewingCos(const float &viewCos)
    {
        if (viewCos > 0.998)
            return 2.5;
        else
            return 4.0;
    }

    // 取出直方图中最大的三个index
    void ORBmatcher::ComputeThreeMaxima(vector<int> *histo, const int L, int &ind1, int &ind2, int &ind3)
    {
        int max1 = 0;
        int max2 = 0;
        int max3 = 0;

        for (int i = 0; i < L; i++)
        {
            const int s = histo[i].size();
            if (s > max1)
            {
                max3 = max2;
                max2 = max1;
                max1 = s;
                ind3 = ind2;
                ind2 = ind1;
                ind1 = i;
            }
            else if (s > max2)
            {
                max3 = max2;
                max2 = s;
                ind3 = ind2;
                ind2 = i;
            }
            else if (s > max3)
            {
                max3 = s;
                ind3 = i;
            }
        }

        if (max2 < 0.1f * (float)max1)
        {
            ind2 = -1;
            ind3 = -1;
        }
        else if (max3 < 0.1f * (float)max1)
        {
            ind3 = -1;
        }
    }

    // 计算描述子距离(hamming距离)
    // Bit set count operation from
    // http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel
    int ORBmatcher::DescriptorDistance(const cv::Mat &a, const cv::Mat &b)
    {
        const int *pa = a.ptr<int32_t>();
        const int *pb = b.ptr<int32_t>();

        int dist = 0;

        for (int i = 0; i < 8; i++, pa++, pb++)
        {
            unsigned int v = *pa ^ *pb;
            v = v - ((v >> 1) & 0x55555555);
            v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
            dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
        }

        return dist;
    }
}
