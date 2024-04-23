#include "MapPoint.h"

#include <mutex>
#include <map>
#include "ORBmatcher.h"

namespace Goudan_SLAM
{
    long unsigned int MapPoint::nNextId = 0;
    mutex MapPoint::mGlobalMutex;

    // 给定坐标和关键帧构造地图点
    MapPoint::MapPoint(const cv::Mat &Pos, KeyFrame *pRefKF, Map *pMap)
        : mnFirstKFid(pRefKF->mnID), mnFirstFrame(pRefKF->mnFrameId), mpMap(pMap), mpRefKF(pRefKF),mbBad(false)
    {
        Pos.copyTo(mWorldPos);
        mNormalVector = cv::Mat::zeros(3, 1, CV_32F);

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

    cv::Mat MapPoint::GetNormal()
    {
        unique_lock<mutex> lock(mMutexPos);
        return mNormalVector.clone();
    }

    KeyFrame *MapPoint::GetReferenceKeyFrame()
    {
        unique_lock<mutex> lock(mMutexFeatures);
        return mpRefKF;
    }

    void MapPoint::AddObservation(KeyFrame *pKF, size_t idx)
    {
        unique_lock<mutex> lock(mMutexFeatures);
        if (mObservations.count(pKF))
            return;

        mObservations[pKF] = idx;
        nObs++;
    }

    int MapPoint::Observations()
    {
        unique_lock<mutex> lock(mMutexFeatures);
        return nObs;
    }

    void MapPoint::EraseObservation(KeyFrame *pKF)
    {
        bool bBad = false;
        {
            unique_lock<mutex> lock(mMutexFeatures);
            if (mObservations.count(pKF))
            {
                int idx = mObservations[pKF];
                nObs--;
                mObservations.erase(pKF);
                // 如果该keyFrame是参考帧，该Frame被删除后重新制定RefFrame
                if (mpRefKF == pKF)
                    mpRefKF = mObservations.begin()->first;

                // 当观测到该点的相机数目少于2时，丢弃该点
                if (nObs <= 2)
                    bBad = true;
            }
        }
        if (bBad)
            SetBadFlag();
    }

    // 告知可以观测到该MapPoint的Frame，该点已被删除
    void MapPoint::SetBadFlag()
    {
        map<KeyFrame *, size_t> obs;
        {
            unique_lock<mutex> lock1(mMutexFeatures);
            unique_lock<mutex> lock2(mMutexPos);
            mbBad = true;
            obs = mObservations;   // 将mObservations 转存到 obs
            mObservations.clear(); // 把mObservations 指向的内存清理
        }
        for (map<KeyFrame *, size_t>::iterator mit = obs.begin(), mend = obs.end(); mit != mend; mit++)
        {
            KeyFrame *pKF = mit->first;
            pKF->EraseMapPointMatch(mit->second);
        }

        mpMap->EraseMapPoint(this);
    }

    // 没有经过MapPointCulling检测的MapPoints
    bool MapPoint::isBad()
    {
        unique_lock<mutex> lock(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPos);
        return mbBad;
    }

    void KeyFrame::EraseMapPointMatch(const size_t &idx)
    {
        unique_lock<mutex> lock(mMutexFeatures);
        mvpMapPoints[idx] = static_cast<MapPoint *>(NULL);
    }

    void KeyFrame::EraseMapPointMatch(MapPoint *pMP)
    {
        int idx = pMP->GetIndexInKeyFrame(this);
        if (idx >= 0)
            mvpMapPoints[idx] = static_cast<MapPoint *>(NULL);
    }

    int MapPoint::GetIndexInKeyFrame(KeyFrame *pKF)
    {
        unique_lock<mutex> lock(mMutexFeatures);
        if (mObservations.count(pKF))
            return mObservations[pKF];
        else
            return -1;
    }

    // 更新平均观测方向以及观测距离范围
    void MapPoint::UpdateNormalAndDepth()
    {
        map<KeyFrame *, size_t> observations;
        KeyFrame *pRefKF;
        cv::Mat Pos;
        {
            unique_lock<mutex> lock1(mMutexFeatures);
            unique_lock<mutex> lock2(mMutexPos);
            if (mbBad)
                return;

            observations = mObservations; // 获得观测到该3d点的所有关键帧
            pRefKF = mpRefKF;             // 观测到该点的参考关键帧
            Pos = mWorldPos.clone();      // 3d点在世界坐标系中的位置
        }

        if (observations.empty())
            return;

        cv::Mat normal = cv::Mat::zeros(3, 1, CV_32F);
        int n = 0;
        for (map<KeyFrame *, size_t>::iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
        {
            KeyFrame *pKF = mit->first;
            cv::Mat Owi = pKF->GetCameraCenter();
            cv::Mat normali = mWorldPos - Owi;
            normal = normal + normali / cv::norm(normali);
            n++;
        }

        cv::Mat PC = Pos - pRefKF->GetCameraCenter(); // 参考关键帧相机指向3D点的向量（在世界坐标系下的表示）
        const float dist = cv::norm(PC);              // 该点到参考关键帧相机的距离
        const int level = pRefKF->mvKeysUn[observations[pRefKF]].octave;
        const float levelScaleFactor = pRefKF->mvScaleFactors[level];
        const int nLevels = pRefKF->mnScaleLevels; // 金字塔层数

        {
            unique_lock<mutex> lock3(mMutexPos);
            // 另见PredictScale函数前的注释
            mfMaxDistance = dist * levelScaleFactor;                             // 观测到该点的距离最大值
            mfMinDistance = mfMaxDistance / pRefKF->mvScaleFactors[nLevels - 1]; // 观测到该点的距离最小值
            mNormalVector = normal / n;                                          // 获得平均的观测方向
        }
    }

    /**
     * @brief 计算具有代表的描述子
     *
     * 由于一个MapPoint会被许多相机观测到，因此在插入关键帧后，需要判断是否更新当前点的最适合的描述子 \n
     * 先获得当前点的所有描述子，然后计算描述子之间的两两距离，最好的描述子与其他描述子应该具有最小的距离中值
     *
     */
    void MapPoint::ComputeDistinctiveDescriptors()
    {
        vector<cv::Mat> vDescriptors;

        std::map<KeyFrame *, size_t> observations;

        {
            unique_lock<mutex> lock1(mMutexFeatures);
            if (mbBad)
                return;
            observations = mObservations;
        }

        if (observations.empty())
            return;

        vDescriptors.reserve(observations.size());

        // 遍历观测到3d点的所有关键帧，获得orb描述子，并插入到vDescriptors中
        for (map<KeyFrame *, size_t>::iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
        {
            KeyFrame *pKF = mit->first;

            // if (!pKF->isBad())
                vDescriptors.push_back(pKF->mDescriptors.row(mit->second));
        }

        if (vDescriptors.empty())
            return;

        // 获得这些描述子两两之间的距离
        const size_t N = vDescriptors.size();

        std::vector<std::vector<float>> Distances;
        Distances.resize(N, vector<float>(N, 0));

        for (size_t i = 0; i < N; i++)
        {
            Distances[i][i] = 0;
            for (size_t j = i + 1; j < N; j++)
            {
                int distij = ORBmatcher::DescriptorDistance(vDescriptors[i], vDescriptors[j]);
                Distances[i][j] = distij;
                Distances[j][i] = distij;
            }
        }

        int BestMedian = INT_MAX;
        int BestIdx = 0;
        for (size_t i = 0; i < N; i++)
        {
            // 第i个描述子到其它所有所有描述子之间的距离
            // vector<int> vDists(Distances[i],Distances[i]+N);
            vector<int> vDists(Distances[i].begin(), Distances[i].end());
            sort(vDists.begin(), vDists.end());

            // 获得中值
            int median = vDists[0.5 * (N - 1)];

            // 寻找最小的中值
            if (median < BestMedian)
            {
                BestMedian = median;
                BestIdx = i;
            }
        }

        {
            unique_lock<mutex> lock(mMutexFeatures);
            // 最好的描述子，该描述子相对于其他描述子有最小的距离中值
            // 简化来讲，中值代表了这个描述子到其它描述子的平均距离
            // 最好的描述子就是和其它描述子的平均距离最小
            mDescriptor = vDescriptors[BestIdx].clone();   
        }
    }
}