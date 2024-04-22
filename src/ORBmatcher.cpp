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
        cout << "matcher: F1 keypoints number :" << F1.mvKeysUn.size() << endl;
        cout << "matcher: F2 keypoints number :" << F2.mvKeysUn.size() << endl;
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
                for (size_t iKF = 0; iKF < vIndicesF.size(); iKF++)
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
